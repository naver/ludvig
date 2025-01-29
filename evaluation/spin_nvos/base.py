import os
import torch
import numpy as np
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms.functional import to_pil_image
from sklearn.linear_model import LogisticRegression
from utils.pca import save_pcas
from utils.image import resize, save_img, image_from_path
from utils.dino import dino_evaluate
from utils.sam import load_sam, sam_evaluate, sam_predict
from utils.visualization import viz_normalization, mask_superposition
from utils.evaluation import segmentation_loop
from utils.data import fetch_data_info
from utils.scribble import load_scribbles, scribble_inverse_rendering
from utils.config import config_to_instance
from evaluation.base import EvaluationBase

class SegmentationBase(EvaluationBase):

    def __init__(
        self,
        *args,
        maskdir,
        model=None,
        segmentation_3d=True,
        sliding_window=None,
        single_view=False,
        use_negative_scribbles=False,
        scribbles_are_masks=False,
        sam_ckpt=None,
        multimask_output=True,
        thresholding="li",
        logreg=None,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.model, self.model_name = None, None
        if model is not None:
            self.model_name = model["name"]
            self.model = config_to_instance(**model)
        self.sam_model = None
        if sam_ckpt is not None:
            self.sam_model = load_sam(sam_ckpt)
        for mask_type in ["binary", "float", "rgb", "comp", "gt_binary", "gt_rgb"]:
            os.makedirs(os.path.join(self.logdir, "masks", mask_type), exist_ok=True)
        self.mask_to_img, self.eval_paths, self.gtpath_from_name, scribble_info = (
            fetch_data_info(maskdir.format(self.scene), self.colmap_cameras)
        )
        cam_name = scribble_info.pop("cam_name")
        self.scribbles_2d = load_scribbles(**scribble_info, size=(self.width, self.height))
        self.scribble_camera = next(
            cam for cam in self.colmap_cameras if cam.image_name == cam_name
        )
        self.sliding_window = sliding_window
        self.segmentation_3d = segmentation_3d
        self.single_view = single_view
        self.use_negative_scribbles = use_negative_scribbles
        self.scribbles_are_masks = scribbles_are_masks
        self.multimask_output = multimask_output
        self.thresholding = thresholding
        self.logreg = logreg
        self.predict_fn = None
        self.gt_img = None
        self.anchor = None
        self.reg_similarities = None
        self.has_saved = False
        self.ev_name = self.eval_paths[0]
        print("Default evaluation:", self.ev_name)

    def __call__(self):
        args = self.hyperparameter_search()
        self.evaluate(*args)

    def hyperparameter_search(self):
        raise NotImplementedError("Base class.")

    def evaluate(*args):
        raise NotImplementedError("Base class.")

    def segment_and_evaluate(
        self,
        features=None,
        k_best=None,
        ev_name=None,
        use_sam=False,
        save=True,
        normalize=True,
        sam_index=None,
        metric="iou",
    ):
        if ev_name is None:
            ev_name = self.ev_name
        img_name = ev_name.split("/")[-1]
        if features is None:
            features = self.features
        if use_sam:
            camera = self.scribble_camera
        else:
            camera = next(
                cam
                for cam in self.colmap_cameras
                if cam.image_name == self.mask_to_img[img_name]
            )
        img_name = os.path.splitext(img_name)[0]

        normalize_rgb = (not use_sam) and not (self.single_view and "sam" in self.model_name.lower())
        rgb_img = image_from_path(self.image_dir, camera.image_name, normalize=normalize_rgb)

        ########## Loading scribbles and computing anchor features ##########
        if not self.segmentation_3d and not self.single_view:
            normalize = False
            feat = self.render_fn(features, self.scribble_camera)
            feat /= feat.norm(dim=0, keepdim=True) + 1e-8
            if self.predict_fn is None:
                self.predict_fn = self.construct_predict_fn(
                    feat, rgb_img, use_sam=use_sam
                )
            if not use_sam:
                feat = self.render_fn(features, camera)
                feat /= feat.norm(dim=0, keepdim=True) + 1e-8
            d, h, w = feat.shape
            feat_ = feat.view(len(feat), -1).T.cpu().numpy()
            anchor = self.predict_fn(feat_)
            self.anchor = torch.from_numpy(anchor).view(h, w).cuda()

        if self.scribbles_are_masks:
            if self.anchor is None:
                print("Uplifting reference mask to be directly used as predicted 3D mask.")
                self.anchor = scribble_inverse_rendering(
                    self.scribbles_2d, self.gaussian, self.scribble_camera
                )
            features = self.anchor

        if use_sam and self.gt_img is not None:
            gt_img = self.gt_img
        elif use_sam:
            print("Multimask output:", self.multimask_output)
            print("Thresholding method:", self.thresholding)
            gt_img = sam_predict(
                model=self.sam_model,
                scribbles_2d=self.scribbles_2d,
                rgb_img=rgb_img,
                multimask_output=self.multimask_output,
                return_mask=self.thresholding,
            )
            gt_img = to_pil_image((gt_img > 0).astype(np.uint8) * 255)
            self.gt_img = gt_img
            gt_img.save(os.path.join(self.logdir, "sam_gt.png"))
        else:
            gt_path = self.gtpath_from_name(ev_name)
            gt_img = Image.open(gt_path)
            gt_img.save(
                os.path.join(self.logdir, "masks", "gt_binary", f"{img_name}.png")
            )

        if self.single_view:
            return self.singleview_evaluate(gt_img, rgb_img, camera, sam_index, k_best)
        #################### Determining 2d similarity ####################
        if self.segmentation_3d:
            anchor = self.render_fn(features.repeat(1, 3), camera)[:1]
        else:
            anchor = self.anchor[None]

        if normalize:
            anchor = viz_normalization(anchor, dim=range(len(anchor.shape)))
        _img_up = resize(anchor, (gt_img.size[1], gt_img.size[0])).squeeze()

        #################### Looping over segmentation thresholds ####################
        best_iou, k_best, mask_best, fpr_tpr = segmentation_loop(
            _img_up, gt_img, k_best, metric=metric
        )
        rbest_iou = int(round(100 * best_iou))

        if not save:
            return best_iou, k_best

        #################### Saving visualizations ####################

        gt_img_arr = np.array(
            gt_img.resize(
                (rgb_img.shape[-1], rgb_img.shape[-2]),
                resample=Image.Resampling.NEAREST,
            )
        )
        gt_img_arr = gt_img_arr // gt_img_arr.max()
        rgb_gt = rgb_img * torch.from_numpy(gt_img_arr[None]).cuda()
        save_img(
            os.path.join(self.logdir, "masks", "gt_rgb", f"{img_name}.jpg"), rgb_gt
        )

        if not self.has_saved and self.features is not None:
            if self.features.shape[-1] > 3:
                save_pcas(self.render_fn(self.features[:, :3], camera), self.logdir)

            os.makedirs(os.path.join(self.logdir, "removal"), exist_ok=True)
            for t in .1*np.arange(1,10):
                mask = (features - features.min()) / (
                    features.max() - features.min()
                ) < t
                # print(f"Selecting {mask.sum()} Gaussians out of {len(mask)} for threshold {t}.")
                self.gaussian.prune_points_noopt(torch.where(mask)[0], backup=True)
                rgb_rm = self.render_rgb(self.colmap_cameras[0])["render"]
                self.gaussian.recover_points()
                save_img(os.path.join(self.logdir, "removal", f"rgb_{t}.png"), rgb_rm)

            if self.reg_similarities is not None:
                to_save = self.reg_similarities.type(torch.float32)[:, None].repeat(1, 3)
                to_save[:, 1:] = 0
                save_img(
                    os.path.join(self.logdir, "regularizer.png"),
                    self.render_fn(to_save, camera),
                )

            self.has_saved = True

        save_img(
            os.path.join(self.logdir, "masks", "float", f"{img_name}.png"),
            anchor.squeeze(),
        )
        save_img(
            os.path.join(self.logdir, "masks", "binary", f"{img_name}_iou{rbest_iou}.png"),
            mask_best,
        )

        mask_arr = (anchor.squeeze() > 1 - k_best / 100).cpu().numpy()
        mask_arr = mask_arr // mask_arr.max()
        mask_diff = mask_superposition(mask_arr, gt_img_arr)
        save_img(
            os.path.join(self.logdir, "masks", "comp", f"{img_name}_iou{rbest_iou}.jpg"),
            anchor.squeeze(),
            mask_diff,
        )

        rgb_mask = rgb_img * (anchor > 1 - k_best / 100)
        save_img(
            os.path.join(self.logdir, "masks", "rgb", f"{img_name}_iou{rbest_iou}.jpg"),
            rgb_mask,
        )

        return best_iou, k_best

    def construct_predict_fn(self, feat, rgb_img, use_sam):
        ### Extracting anchor features ###
        scribbles = self.scribbles_2d
        if use_sam:
            print("Replacing scribbles with SAM mask.")
            scribbles_sam = sam_predict(self.sam_model, scribbles, rgb_img)
            gt_img_pil = to_pil_image((scribbles_sam > 0).astype(np.uint8) * 255)
            self.gt_img = gt_img_pil
            gt_img_pil.save(os.path.join(self.logdir, "sam_gt.png"))
            if self.use_negative_scribbles:
                print("Using SAM for positives and negative scribbles for negatives.")
                mask = (scribbles_sam > 0).squeeze() + (scribbles[1] > 0).cpu().numpy()
                feat_ = feat[:, mask].T
                scribbles_ = (scribbles_sam[mask] > 0).squeeze().flatten()
            else:
                feat_ = feat.view(len(feat), -1).T
                scribbles_ = (scribbles_sam > 0).squeeze().flatten()
        elif len(scribbles) > 1:
            print("Running logistic regression with negatives scribbles.")
            mask = (scribbles[0] > 0) + (scribbles[1] > 0)
            feat_ = feat[:, mask].T
            scribbles_ = scribbles[0][mask].cpu().numpy()
        else:
            feat_ = feat.view(len(feat), -1).T
            scribbles_ = scribbles[0].flatten().cpu().numpy()
        if self.logreg:
            logreg = LogisticRegression(
                C=self.logreg, max_iter=1000, class_weight="balanced"
            ).fit(feat_.cpu().numpy(), scribbles_.astype(int))
            return lambda feat: logreg.predict_proba(feat)[:, 1]
        return lambda feat: F.cosine_simiarity(
            feat_.mean(0, keepdim=True), feat, dim=-1
        )

    def singleview_evaluate(self, gt_img, rgb_img, camera, sam_index, k_best):
        assert self.model is not None
        if "dino" in self.model_name.lower():
            rgb_img_scribbles = image_from_path(
                self.image_dir, self.scribble_camera.image_name, normalize=True
            )
            best_iou, k_best, self.predict_fn = dino_evaluate(
                self.model,
                self.predict_fn,
                self.logreg,
                gt_img,
                self.logdir,
                self.scribbles_2d[0],
                rgb_img,
                rgb_img_scribbles,
                sliding_window=self.sliding_window,
                k_best=k_best,
            )
        else:
            scribbles_2d = self.render_fn(self.anchor.repeat(1, 3), camera)[0]
            best_iou, k_best = sam_evaluate(
                self.model,
                rgb_img,
                scribbles_2d,
                gt_img=gt_img,
                eval_dir=self.logdir,
                multimask_output=False,
                sam_index=sam_index,
                k_best=k_best,
            )
        return best_iou, k_best
