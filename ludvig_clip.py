import sys
import numpy as np
import torch
import os
from time import time
from tqdm import tqdm
import torch.nn.functional as F
from argparse import ArgumentParser
from skimage import filters
from sklearn.decomposition import PCA
from torchvision.transforms.functional import to_pil_image

from diffusion.clip import GraphDiffusionCLIP
from utils.image import save_img, image_from_path
from utils.sam import load_sam, sam_predict
from utils.visualization import mask_superposition, viz_normalization
from utils.evaluation import iou
from clip_utils.lerf import load_gt_lerfdata, is_in_box
from clip_utils.visualization import heatmap_fn, visualize_loc
from clip_utils.openclip_encoder import OpenCLIPNetwork
from ludvig_base import LUDVIGBase, reproducibility


class LUDVIGCLIP(LUDVIGBase):

    def __init__(self, cfg) -> None:
        super().__init__(cfg)
        self.load_clip = cfg.clip_features
        self.load_dino = cfg.dino_features
        self.img_height, self.img_width = cfg.height, cfg.width
        if cfg.load_ply:
            print("Loading gaussians from", cfg.load_ply)
            self.gaussian.load_ply(
                os.path.join(self.config.dst_dir, self.scene, cfg.load_ply)
            )
        self.clip_model = OpenCLIPNetwork("cuda")
        self.gt_ann, self.img_paths = load_gt_lerfdata(
            os.path.join(self.config.evaluate, self.scene)
        )
        self.prompts = sorted(
            set([k for img_ann in self.gt_ann.values() for k in img_ann.keys()])
        )
        print("Prompts:", self.prompts)
        print(
            "Number of prompts across test images:",
            sum([len(d) for d in self.gt_ann.values()]),
        )
        self.prompt2idx = {k: i for i, k in enumerate(self.prompts)}
        self.clip_model.set_positives(self.prompts)
        self.sam_model = None
        if self.config.get("sam_ckpt", None):
            self.sam_model = load_sam(self.config.sam_ckpt)
        self.use_sam = not args.no_sam and self.sam_model is not None
        self.use_diffusion = not args.no_diffusion and self.config.get("diffusion_cfg", None) is not None
        self.graph_diffusion = None
        self.relev = self.compute_relevancies()
        self.n_runs = cfg.n_runs
        self.save = not cfg.no_saving
        self.pbar = None

    def compute_relevancies(self, sem=None):
        """Compute per-Gaussian relevancy score with text queries based on 3D CLIP features."""
        if sem is None:
            sem = torch.from_numpy(np.load(self.load_clip)).cuda()
        sem /= torch.norm(sem, dim=-1, keepdim=True) + 1e-6
        return self.clip_model.get_max_across(sem)

    def evaluate_base(self, save=True):
        """Evaluate 3D relevancies `self.relev` on LERF localization and segmentation."""
        return self.evaluate_lerf(
            save_loc=save,
            save_seg=save * (not self.use_diffusion),
            use_sam=self.use_sam,
            **self.config.get("evaluation", dict()),
        )

    def evaluate_diffusion(self, save=True):
        """Evaluate 3D relevancies `self.relev` on LERF localization and segmentation."""
        return self.evaluate_lerf(
            use_sam=self.use_sam,
            save_seg=save,
            **self.config.get("evaluation_diffusion", dict()),
        )

    def run_diffusion(self):
        """Run graph diffusion to refine 3D CLIP relevancy scores."""
        t0 = time()
        self.graph_diffusion = GraphDiffusionCLIP(
            gaussian=self.gaussian,
            render_fn=self.render,
            cameras=self.colmap_cameras,
            load_dino=self.load_dino,
            logdir=self.logdir,
            eps=1e-6,
            relev=self.relev,
            **self.config.diffusion_cfg,
        )
        self.relev = self.graph_diffusion().T
        print(
            f"Total time for graph initialization + graph diffusion: {round(time()-t0)} seconds."
        )
        self.visualize_regularizer()

    def visualize_regularizer(self):
        """Save a visualization of the graph diffusion regularization term."""
        trace_names = dict(
            figurines="frame_00195",
            ramen="frame_00006",
            teatime="frame_00129",
            waldo_kitchen="frame_00053",
        )
        cam = next(
            cam
            for cam in self.colmap_cameras
            if cam.image_name == trace_names[self.scene]
        )
        regularizer = heatmap_fn(
            self.render_rgb(cam)["render"],
            self.render(
                self.graph_diffusion.compute_regularizer().max(dim=1, keepdim=True).values.repeat(1, 3),
                cam,
            ),
        )
        save_img(os.path.join(self.logdir, "regularizer.jpg"), regularizer)

    def save_features(self, video=False):
        """Save visualizations of 3D CLIP features, DINOv2 features, and relevancy with each text query."""
        clipf = torch.from_numpy(np.load(self.load_clip)).cuda()
        clipf /= torch.norm(clipf, dim=1, keepdim=True) + 1e-8
        clipf = torch.from_numpy(
            PCA(n_components=3).fit_transform(clipf.cpu().numpy())
        ).cuda()
        dinof = torch.from_numpy(np.load(self.load_dino)).cuda()[:, :3]
        interpolate = 10
        if not video:
            interpolate = 0
            cameras = [cam for cam in self.colmap_cameras if cam.image_name in self.gt_ann]
        print("Saving feature visualizations...")
        pbar = tqdm(
            zip(self.prompts, self.relev),
            total=len(self.prompts),
            bar_format="{n_fmt}/{total_fmt} Saving for {desc}",
        )
        for i, (p, rel) in enumerate(pbar):
            if video:
                cameras = self.select_cameras(rel[:, None].repeat(1, 3), k=10)
            pbar.set_description(p)
            rel_ = rel.cpu().numpy()
            if np.any(rel_ > 0):
                thres = filters.threshold_otsu(rel_[rel_ > 0])
                rel[rel < thres] = 0
                self.save_images(
                    rel[:, None].repeat(1, 3),
                    "rel/" + p.replace(" ", "_"),
                    joint_fn=heatmap_fn,
                    cameras=cameras,
                    text=p,
                    font_size=50,
                    interpolate=interpolate,
                )
            if not video and i > 1:
                continue
            self.save_images(
                clipf,
                "clip/" + p.replace(" ", "_") * video,
                pca=True,
                cameras=cameras,
                interpolate=interpolate,
            )
            self.save_images(
                dinof,
                "dino/" + p.replace(" ", "_") * video,
                pca=True,
                cameras=cameras,
                interpolate=interpolate,
            )

    def evaluate_lerf(
        self,
        save_loc=False,
        save_seg=False,
        use_sam=False,
        **kwargs,
    ):
        """Evaluate on each test image in LERF."""
        ious = []
        acc_num = 0
        coverage = []
        self.skipped = []
        t0 = time()
        for ev in self.gt_ann.keys():
            camera = next(cam for cam in self.colmap_cameras if cam.image_name == ev)
            rel = self.render(self.relev.T, camera)
            img = image_from_path(os.path.join(self.colmap_dir, "images"), ev)
            if use_sam:
                img_arr = np.asarray(to_pil_image(img.cpu()))
                self.sam_model.set_image(img_arr)
            img_ann = self.gt_ann[ev]
            logdir_loc, logdir_seg = None, None
            if save_loc:
                logdir_loc = os.path.join(self.logdir, "localization", ev)
                os.makedirs(logdir_loc, exist_ok=True)
            if save_seg:
                name = "masks" + "_diffusion" * self.use_diffusion + "_sam" * self.use_sam
                logdir_seg = os.path.join(self.logdir, name, ev)
                os.makedirs(os.path.join(logdir_seg, "comp"), exist_ok=True)
                os.makedirs(os.path.join(logdir_seg, "ground_truth"), exist_ok=True)
                os.makedirs(os.path.join(logdir_seg, "heatmap"), exist_ok=True)
            _iou, acc, cov = self.evaluate_relevancy(
                rel,
                img_ann,
                logdir_loc=logdir_loc,
                logdir_seg=logdir_seg,
                image=img,
                sam_model=self.sam_model if use_sam else None,
                **kwargs,
            )
            ious.extend(_iou)
            coverage.extend(cov)
            acc_num += acc
        total_bboxes = 0
        for img_ann in self.gt_ann.values():
            total_bboxes += len(list(img_ann.keys()))
        acc = 100 * acc_num / total_bboxes
        mean_iou = 100 * sum(ious) / len(ious)
        if self.pbar is not None:
            self.pbar.update(1)
            self.pbar.set_description(f"Localization: {round(acc, 1)} - IoU: {round(mean_iou, 1)}")
        # if len(self.skipped):
        #    print(f"Prompts {set(self.skipped)} scored zero IoU on some views as relevancies were all zero.")
        return mean_iou, acc, time() - t0

    def select_cameras(self, rel, k=5):
        """Select cameras for which projected relevancy is the highest for making object-centered videos."""
        mean_relevancies = []
        for cam in self.colmap_cameras:
            feat = self.render(rel, cam)
            feat_ = feat[0].cpu().numpy()
            if (feat_ > 0).sum() < 100:
                mean_relevancies.append(0)
                continue
            mean_relevancies.append((feat * (feat / feat.max() > 0.6)).mean().item())
        cameras = [self.colmap_cameras[i] for i in sorted(np.argsort(mean_relevancies)[-k:])]
        print(f"Keeping {len(cameras)} cameras.")
        return cameras

    def evaluate_relevancy(
        self,
        relevancies,
        img_ann,
        thres="otsu",
        cutoff=0,
        smooth=None,
        logdir_loc=None,
        logdir_seg=None,
        image=None,
        sam_model=None,
        npr=20,
        viz_normalize=True,
    ):
        """
        Evaluate LERF localization and segmentation on one test image.

        Args:
            relevancies (torch.Tensor): 2D relevancies for each prompt, of shape (P, H, W)
            img_ann (dict): Image annotations (bounding box and segmentation mask). Keys correspond to prompts.
            thres (str, optional): Thresholding method for segmentation ("otsu", "li", or "yen").
            cutoff (float, optional): A lower cutoff value for thresholding, normalized by the maximum relevancy.
            smooth (int, optional): Size of the smoothing kernel for applying average pooling to the relevancy maps.
            logdir_loc (str, optional): Directory path to save localization visualizations.
            logdir_seg (str, optional): Directory path to save segmentation visualizations.
            image (torch.Tensor or np.ndarray, optional): The RGB image.
            sam_model (object, optional): SAM instance. If None, threshold-based segmentation is used.
            npr (int, optional): Number of SAM mask predictions to average over.
            viz_normalize (bool, optional): Whether to normalize the relevancy map for visualization.
        """

        ious = []
        acc = 0
        coverage = []
        for p, v in img_ann.items():
            k = self.prompt2idx[p]
            rel = relevancies[k]
            if smooth:
                kernel = torch.ones((1, 1, smooth, smooth), device="cuda") / (smooth**2)
                rel = F.conv2d(rel[None, None], kernel, padding="same").squeeze(0).squeeze(0)
            if not torch.any(rel > 0):
                self.skipped.append(p)
                ious.append(0)
                continue

            # Localization
            coord = torch.argwhere(rel == rel.max()).cpu().numpy()[:, ::-1]
            acc += is_in_box(coord, v["bboxes"])

            # Segmentation
            method = dict(
                otsu=filters.threshold_otsu,
                li=filters.threshold_li,
                yen=filters.threshold_yen,
            )[thres]
            t = method(rel[rel / rel.max() > cutoff].cpu().numpy())
            if sam_model is not None:
                qmax = 1 - (rel > t).type(torch.float32).mean()
                mask_pred = rel >= torch.quantile(rel, qmax)
                mask_pred = sam_predict(
                    sam_model,
                    [mask_pred],
                    npo=3,
                    npr=npr,
                    multimask_output=False,
                    return_mask=thres,
                ).squeeze()
                mask_pred = (mask_pred > 0).astype(int)
            else:
                mask_pred = (rel > t).cpu().numpy().astype(int)
            mask_gt = v["mask"].astype(np.uint8)

            if viz_normalize:
                rel[rel > 0] = viz_normalization(rel[rel > 0], clip=5)
            heatmap = heatmap_fn(
                image,
                rel[None].repeat(3, 1, 1),
                ~torch.from_numpy(mask_pred.astype(bool))[None].repeat(3, 1, 1).cuda(),
            )
            if logdir_loc is not None:
                visualize_loc(
                    heatmap.permute(1, 2, 0).cpu().numpy(),
                    coord,
                    v["bboxes"],
                    os.path.join(logdir_loc, "{}.jpg"),
                    p,
                )

            if logdir_seg is not None:
                mask_diff = mask_superposition(mask_pred, mask_gt)
                if image is not None:
                    img_mask = ~torch.from_numpy((mask_pred + mask_gt).astype(bool))
                    mask_diff[:, img_mask] = 0.3 * image[:, img_mask].cpu()
                save_img(os.path.join(logdir_seg, "comp", p + ".jpg"), mask_diff, text=p)

                mask_gt_colored = torch.from_numpy(mask_gt[None].repeat(3, 0)).type(torch.float32)
                if image is not None:
                    img_mask = ~mask_gt_colored.type(torch.bool)
                    mask_gt_colored[:, img_mask[0]] = 0.3 * image[:, img_mask[0]].cpu()
                save_img(
                    os.path.join(logdir_seg, "ground_truth", p + ".jpg"),
                    mask_gt_colored,
                    text=p,
                )
                save_img(os.path.join(logdir_seg, "heatmap", p + ".jpg"), heatmap, text=p)

            coverage.append(mask_pred.sum() / mask_gt.sum())
            ious.append(iou(mask_gt, mask_pred, class_label=1))

        return ious, acc, coverage

    def evaluate(self):
        """Evaluate on LERF with and without graph diffusion, averaged over multiple runs when using SAM."""
        dst_file = "iou_sam.txt" if self.use_sam else "iou.txt"
        dstf = open(os.path.join(self.logdir, dst_file), "w")

        if self.use_sam:
            print(f"\nAveraging SAM evaluation results across {self.n_runs} runs.")
        print("\n--------------- Evaluating without graph diffusion ---------------")
        if self.use_sam:
            self.pbar = tqdm(
                total=self.n_runs, bar_format="{n_fmt}/{total_fmt} | {desc}"
            )
            iou_uplift, acc_uplift, times = list(
                zip(*[
                    self.evaluate_base(save=(i == 0) * (~self.use_diffusion) * self.save)
                    for i in range(self.n_runs)
                ])
            )
            self.pbar.close()
            inference_time = times[-1]
            acc_uplift = sum(acc_uplift) / len(acc_uplift)
            avg_iou = round(sum(iou_uplift) / len(iou_uplift), 1)
            iou_uplift = [round(x, 1).item() for x in iou_uplift]
            msg = f"IoU uplifted ({self.n_runs} indep. runs): {iou_uplift} - Mean: {avg_iou}"
        else:
            iou_uplift, acc_uplift, inference_time = self.evaluate_base(save=self.save)
            msg = f"IoU uplifted: {round(iou_uplift,1)}"
        msg_loc = f"Localization accuracy: {round(acc_uplift,1)}"
        if self.use_sam or not self.save:
            print(f"Inference times: {round(inference_time, 1)} seconds.")
        print(msg_loc)
        dstf.write(msg_loc + "\n")
        print(msg + "\n")
        dstf.write(msg + "\n")
        if self.use_diffusion:
            print("--------------- Evaluating with graph diffusion ---------------")
            self.run_diffusion()
            if self.use_sam:
                self.pbar = tqdm(
                    total=self.n_runs, bar_format="{n_fmt}/{total_fmt} | {desc}"
                )
                iou_diffusion = [
                    round(self.evaluate_diffusion(save=(i == 0) * self.save)[0], 1).item()
                    for i in range(self.n_runs)
                ]
                self.pbar.close()
                avg_iou = round(sum(iou_diffusion) / len(iou_diffusion), 1)
                msg = f"IoU diffusion ({self.n_runs} indep. runs): {iou_diffusion} - Mean: {avg_iou}"
            else:
                iou_diffusion = round(self.evaluate_diffusion()[0], 1)
                msg = f"IoU diffusion: {iou_diffusion}"
            print(msg + "\n")
            dstf.write(msg + "\n")
        dstf.close()
        self.save_features()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--gs_source", type=str, required=True)  # gs ply or obj file?
    parser.add_argument("--colmap_dir", type=str, required=True)  #
    parser.add_argument("--config", type=str, required=True)  #
    parser.add_argument("--dino_features", type=str)  #
    parser.add_argument("--clip_features", type=str)  #
    parser.add_argument("--height", type=int)  #
    parser.add_argument("--width", type=int)  #
    parser.add_argument("--load_ply", type=str)  #
    parser.add_argument("--no_sam", action="store_true")  #
    parser.add_argument("--no_diffusion", action="store_true")  #
    parser.add_argument("--tag", type=str)  #
    parser.add_argument("--n_runs", type=int, default=3)  #
    parser.add_argument("--no_saving", action="store_true")  #

    args = parser.parse_args()
    reproducibility(0)
    model = LUDVIGCLIP(args)
    model.evaluate()
    sys.exit()
