import os


def fetch_data_info(evaluate, colmap_cameras, use_negatives=False):
    if "spin-nerf" in evaluate.lower():
        return spinnerf_paths(evaluate, colmap_cameras)
    return nvos_paths(evaluate, colmap_cameras, use_negatives)


def spinnerf_paths(evaluate, colmap_cameras):
    fnames = os.listdir(evaluate)
    mask_names = [
        x.replace("_cutout", "").replace("_pseudo", "")
        for x in fnames
        if "json" not in x
    ]
    mask_names = sorted(set(mask_names))
    img_names = sorted([cam.image_name for cam in colmap_cameras])
    if "orchids" in evaluate:
        img_names = img_names[:14] + img_names[15:]
    if "truck" in evaluate or "lego" in evaluate:
        img_names = [mask.split("_")[1].split(".")[0] for mask in mask_names]
    mask_to_img = {k: v for k, v in zip(mask_names, img_names)}
    eval_paths = [os.path.join(evaluate, mask_name) for mask_name in mask_names]
    gt_path = lambda ev: ev
    scribble_info = dict(
        path=os.path.join(evaluate, mask_names[0]),
        cam_name=img_names[0],
        tags=[""],
    )
    return mask_to_img, eval_paths, gt_path, scribble_info


def nvos_paths(evaluate, colmap_cameras, use_negatives):
    img_names = sorted([cam.image_name for cam in colmap_cameras])
    mask_to_img = {name: name for name in img_names}
    mask_path = evaluate.format("masks")
    eval_paths = sorted(
        [
            os.path.join(mask_path, ev.split(".")[0])
            for ev in os.listdir(mask_path)
            if "mask" not in ev
        ]
    )
    gt_path = lambda ev: ev + "_mask.png"
    scribble_info = dict(
        path=evaluate.format("scribbles"),
        cam_name=os.path.splitext(os.listdir(evaluate.format("reference_image"))[0])[0],
        tags=["pos"] + ["neg"] * use_negatives,
    )
    return mask_to_img, eval_paths, gt_path, scribble_info
