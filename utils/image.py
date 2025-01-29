import os
import torch
import numpy as np
from torchvision.transforms import Normalize
from PIL import Image, ImageDraw, ImageFont
from gaussiansplatting.utils.general_utils import PILtoTorch


MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


def image_from_path(image_folder, cam_name, resize_down=1600, normalize=False):
    ext = os.path.splitext(os.listdir(image_folder)[0])[1]
    image_path = os.path.join(image_folder, cam_name + ext)
    image = Image.open(image_path)
    orig_w, orig_h = image.size
    if resize_down and orig_w > resize_down:
        global_down = orig_w / resize_down
    else:
        global_down = 1
    resolution = (int(orig_w / global_down), int(orig_h / global_down))
    image = PILtoTorch(image, resolution)
    if normalize:
        image = Normalize(MEAN, STD)(image)
    return image.cuda()


def resize(mask, size):
    return torch.nn.functional.interpolate(
        input=mask[None],
        size=size,
        mode="bilinear",
    ).squeeze(0)


def to_pil(img):
    img = img.type(torch.float32)
    img = (img - img.min()) / (img.max() - img.min())
    if len(img.shape) == 3 and len(img) == 3:
        img = img.moveaxis(0, -1)
    img = (img * 255).to(torch.uint8).cpu()
    img = img.numpy().astype(np.uint8)
    img = Image.fromarray(img)
    return img


def save_img(dst_path, *imgs, text=None, font_path="times-new-roman.ttf", font_size=80):
    pil_imgs = []
    for img in imgs:
        if isinstance(img, torch.Tensor):
            img = to_pil(img)
        pil_imgs.append(img)
    if len(pil_imgs) > 1:
        assert len(pil_imgs) <= 3, "Not handling more than 3 images for now."
        if len(pil_imgs) == 3:
            total_width = max(pil_imgs[0].width + pil_imgs[1].width, pil_imgs[2].width)
            positions = [(0, 0), (pil_imgs[0].width, 0), (0, pil_imgs[0].height)]
        else:
            total_width = max(pil_imgs[0].width, pil_imgs[1].width)
            positions = [(0, 0), (0, pil_imgs[0].height)]
        max_height = pil_imgs[-1].height + pil_imgs[-2].height
        new_img = Image.new("RGB", (total_width, max_height), "white")
        for i, (img, p) in enumerate(zip(pil_imgs, positions)):
            new_img.paste(img, p)
    else:
        total_width = pil_imgs[0].width
        new_img = pil_imgs[0]

    if text:
        draw = ImageDraw.Draw(new_img)
        if font_path and os.path.exists(font_path):
            font = ImageFont.truetype(font_path, font_size)
        else:
            font = ImageFont.load_default()
        text_bbox = font.getbbox(text)
        text_width = text_bbox[2] - text_bbox[0]
        x_position = total_width - text_width - 40
        y_position = 40
        draw.text((x_position, y_position), text, font=font, fill="white")

    new_img.save(dst_path)
    return
