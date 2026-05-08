import os
import io
import torch
import numpy as np
from torchvision import transforms
import torchvision.transforms.functional as TF
from PIL import Image, ImageFilter, ImageEnhance
import random


def set_random_seed(seed=0):
    torch.manual_seed(seed + 0)
    torch.cuda.manual_seed(seed + 1)
    torch.cuda.manual_seed_all(seed + 2)
    np.random.seed(seed + 3)
    torch.cuda.manual_seed_all(seed + 4)
    random.seed(seed + 5)


def transform_img(image, target_size=512):
    tform = transforms.Compose(
        [
            transforms.Resize(target_size),
            transforms.CenterCrop(target_size),
            transforms.ToTensor(),
        ]
    )
    image = tform(image)
    return 2.0 * image - 1.0


def latents_to_imgs(pipe, latents):
    x = pipe.decode_image(latents)
    x = pipe.torch_to_numpy(x)
    x = pipe.numpy_to_pil(x)
    return x


def _clip_uint8_image_array(arr):
    return np.clip(arr, 0, 255).astype(np.uint8)


def _ensure_rgb_pil(img):
    if not isinstance(img, Image.Image):
        img = Image.fromarray(_clip_uint8_image_array(np.asarray(img)))
    if img.mode != "RGB":
        img = img.convert("RGB")
    return img


def _sample_uniform_if_needed(seed, fixed_value, min_value, max_value):
    """Return fixed_value if provided; otherwise sample uniformly from [min_value, max_value]."""
    if fixed_value is not None:
        return float(fixed_value)
    if min_value is not None and max_value is not None:
        return float(np.random.uniform(float(min_value), float(max_value)))
    return None


def apply_rst_attack(
    img,
    seed=0,
    rotation_degree=None,
    random_rotation_degree=None,
    scale=None,
    scale_min=None,
    scale_max=None,
    translate_x=0.0,
    translate_y=0.0,
    shear=0.0,
    resized_crop_ratio=None,
    crop_side_ratio=None,
    order="affine_then_crop",
    fill=0,
):
    """
    Apply an RST-style geometric attack to a PIL image.

    This function follows the same practical attack family used by recent
    diffusion-watermark papers and repositories:
      - affine rotation/scale/translation using torchvision.functional.affine;
      - crop-and-scale using RandomResizedCrop-style random crop followed by
        resizing back to the original resolution.

    Parameters
    ----------
    rotation_degree:
        Fixed rotation angle in degrees. Positive values follow torchvision's
        convention.
    random_rotation_degree:
        If provided and rotation_degree is None, sample the angle from
        [-random_rotation_degree, random_rotation_degree].
    scale:
        Fixed affine scale. 1.0 means no scaling. Values >1 zoom in; values <1
        zoom out in torchvision's affine convention.
    scale_min, scale_max:
        If scale is None and both are provided, sample scale uniformly.
    translate_x, translate_y:
        Translation ratios relative to image width/height. For example,
        translate_x=0.05 moves 5% of the image width.
    resized_crop_ratio:
        Area ratio for crop-and-scale. 0.75 means crop 75% of image area and
        resize it back to the original image size. This matches the
        RandomResizedCrop scale semantics used by WAVE/MaXsive-style attacks.
    crop_side_ratio:
        Side-length ratio for crop-and-scale. 0.75 means crop a square with
        side length 75% of the original side length and resize it back.
        If both resized_crop_ratio and crop_side_ratio are set,
        resized_crop_ratio has priority.
    order:
        "affine_then_crop" or "crop_then_affine".
    fill:
        Fill value for areas outside the image after affine transform.
    """
    img = _ensure_rgb_pil(img)
    set_random_seed(seed)

    if order not in ["affine_then_crop", "crop_then_affine"]:
        raise ValueError("order must be 'affine_then_crop' or 'crop_then_affine'.")

    width, height = img.size

    sampled_angle = rotation_degree
    if sampled_angle is None and random_rotation_degree is not None:
        max_angle = abs(float(random_rotation_degree))
        sampled_angle = float(np.random.uniform(-max_angle, max_angle))
    if sampled_angle is None:
        sampled_angle = 0.0

    sampled_scale = _sample_uniform_if_needed(seed, scale, scale_min, scale_max)
    if sampled_scale is None:
        sampled_scale = 1.0
    sampled_scale = float(sampled_scale)
    if sampled_scale <= 0:
        raise ValueError("RST scale must be positive.")

    translate_px = [int(round(float(translate_x) * width)), int(round(float(translate_y) * height))]
    sampled_shear = float(shear) if shear is not None else 0.0

    def do_affine(x):
        return TF.affine(
            x,
            angle=float(sampled_angle),
            translate=translate_px,
            scale=sampled_scale,
            shear=sampled_shear,
            interpolation=TF.InterpolationMode.BILINEAR,
            fill=fill,
        )

    def do_crop_scale(x):
        if resized_crop_ratio is None and crop_side_ratio is None:
            return x

        w, h = x.size
        if resized_crop_ratio is not None:
            # torchvision RandomResizedCrop uses area ratio.
            area_ratio = float(resized_crop_ratio)
            if not (0 < area_ratio <= 1):
                raise ValueError("resized_crop_ratio must be in (0, 1].")
            i, j, crop_h, crop_w = transforms.RandomResizedCrop.get_params(
                x,
                scale=(area_ratio, area_ratio),
                ratio=(1.0, 1.0),
            )
        else:
            # Direct side-length ratio.
            side_ratio = float(crop_side_ratio)
            if not (0 < side_ratio <= 1):
                raise ValueError("crop_side_ratio must be in (0, 1].")
            crop_w = max(1, int(round(w * side_ratio)))
            crop_h = max(1, int(round(h * side_ratio)))
            j = np.random.randint(0, w - crop_w + 1)
            i = np.random.randint(0, h - crop_h + 1)

        return TF.resized_crop(
            x,
            top=i,
            left=j,
            height=crop_h,
            width=crop_w,
            size=[h, w],
            interpolation=TF.InterpolationMode.BILINEAR,
        )

    if order == "affine_then_crop":
        img = do_affine(img)
        img = do_crop_scale(img)
    else:
        img = do_crop_scale(img)
        img = do_affine(img)

    return img


def get_attack_summary(args):
    """Return a compact dictionary of active image attacks for logging."""
    keys = [
        "jpeg_ratio",
        "random_crop_ratio",
        "random_drop_ratio",
        "gaussian_blur_r",
        "median_blur_k",
        "resize_ratio",
        "gaussian_std",
        "sp_prob",
        "brightness_factor",
        "rotation_degree",
        "rst_rotation_degree",
        "rst_random_rotation_degree",
        "rst_scale",
        "rst_scale_min",
        "rst_scale_max",
        "rst_translate_x",
        "rst_translate_y",
        "rst_shear",
        "rst_resized_crop_ratio",
        "rst_crop_side_ratio",
        "rst_order",
        "vae_attack",
        "vae_model",
        "vae_quality",
        "regen_attack",
        "regen_model_path",
        "regen_noise_step",
        "regen_denoise_steps",
        "regen_guidance_scale",
        "regen_use_prompt",
        "regen_start_step_mode",
        "advanced_attack_order",
    ]
    summary = {}
    for key in keys:
        if hasattr(args, key):
            value = getattr(args, key)
            if value is not None and value is not False:
                summary[key] = value
    return summary


def image_distortion(img, seed, args, advanced_attackers=None, prompt=""):
    """Apply enabled image attacks in a reproducible order.

    Basic pixel/geometric attacks are applied first. Optional advanced attacks
    such as VAE compression and diffusion regeneration are initialized once in
    run_gaussian_shading.py and passed through advanced_attackers.
    """
    img = _ensure_rgb_pil(img)
    advanced_attackers = advanced_attackers or {}

    if args.jpeg_ratio is not None:
        # Use an in-memory buffer to avoid temporary-file conflicts in long runs.
        buffered = io.BytesIO()
        img.save(buffered, format="JPEG", quality=int(args.jpeg_ratio))
        buffered.seek(0)
        img = Image.open(buffered).convert("RGB")

    if args.random_crop_ratio is not None:
        # Original ANCHOR random crop attack: keep a random crop at its original
        # location and zero out the rest. This is not crop-and-scale.
        set_random_seed(seed)
        arr = np.array(img)
        height, width, _ = arr.shape
        new_width = int(width * args.random_crop_ratio)
        new_height = int(height * args.random_crop_ratio)
        start_x = np.random.randint(0, width - new_width + 1)
        start_y = np.random.randint(0, height - new_height + 1)
        end_x = start_x + new_width
        end_y = start_y + new_height
        padded_image = np.zeros_like(arr)
        padded_image[start_y:end_y, start_x:end_x] = arr[start_y:end_y, start_x:end_x]
        img = Image.fromarray(padded_image)

    if args.random_drop_ratio is not None:
        set_random_seed(seed)
        arr = np.array(img)
        height, width, _ = arr.shape
        new_width = int(width * args.random_drop_ratio)
        new_height = int(height * args.random_drop_ratio)
        start_x = np.random.randint(0, width - new_width + 1)
        start_y = np.random.randint(0, height - new_height + 1)
        arr[start_y:start_y + new_height, start_x:start_x + new_width] = 0
        img = Image.fromarray(arr)

    if args.resize_ratio is not None:
        img_shape = np.array(img).shape
        resize_size = int(img_shape[0] * args.resize_ratio)
        img = transforms.Resize(size=resize_size)(img)
        img = transforms.Resize(size=img_shape[0])(img)

    if args.gaussian_blur_r is not None:
        img = img.filter(ImageFilter.GaussianBlur(radius=args.gaussian_blur_r))

    if args.median_blur_k is not None:
        img = img.filter(ImageFilter.MedianFilter(args.median_blur_k))

    if args.gaussian_std is not None:
        set_random_seed(seed)
        img_shape = np.array(img).shape
        g_noise = np.random.normal(0, args.gaussian_std, img_shape) * 255
        img = Image.fromarray(_clip_uint8_image_array(np.array(img).astype(np.float32) + g_noise))

    if args.sp_prob is not None:
        set_random_seed(seed)
        arr = np.array(img)
        h, w, c = arr.shape
        prob_zero = args.sp_prob / 2
        prob_one = 1 - prob_zero
        rdn = np.random.rand(h, w, c)
        arr = np.where(rdn > prob_one, np.zeros_like(arr), arr)
        arr = np.where(rdn < prob_zero, np.ones_like(arr) * 255, arr)
        img = Image.fromarray(arr.astype(np.uint8))

    if args.brightness_factor is not None:
        img = transforms.ColorJitter(brightness=args.brightness_factor)(img)

    # Existing single rotation flag. Kept for compatibility with older scripts.
    if hasattr(args, "rotation_degree") and args.rotation_degree is not None:
        img = TF.rotate(
            img,
            angle=float(args.rotation_degree),
            interpolation=TF.InterpolationMode.BILINEAR,
            fill=0,
        )

    # RST attack block. It can be used alone or together with the older attacks.
    has_rst_attack = any([
        hasattr(args, "rst_rotation_degree") and args.rst_rotation_degree is not None,
        hasattr(args, "rst_random_rotation_degree") and args.rst_random_rotation_degree is not None,
        hasattr(args, "rst_scale") and args.rst_scale is not None,
        hasattr(args, "rst_scale_min") and args.rst_scale_min is not None,
        hasattr(args, "rst_scale_max") and args.rst_scale_max is not None,
        hasattr(args, "rst_translate_x") and args.rst_translate_x not in [None, 0, 0.0],
        hasattr(args, "rst_translate_y") and args.rst_translate_y not in [None, 0, 0.0],
        hasattr(args, "rst_shear") and args.rst_shear not in [None, 0, 0.0],
        hasattr(args, "rst_resized_crop_ratio") and args.rst_resized_crop_ratio is not None,
        hasattr(args, "rst_crop_side_ratio") and args.rst_crop_side_ratio is not None,
    ])

    if has_rst_attack:
        img = apply_rst_attack(
            img=img,
            seed=seed,
            rotation_degree=getattr(args, "rst_rotation_degree", None),
            random_rotation_degree=getattr(args, "rst_random_rotation_degree", None),
            scale=getattr(args, "rst_scale", None),
            scale_min=getattr(args, "rst_scale_min", None),
            scale_max=getattr(args, "rst_scale_max", None),
            translate_x=getattr(args, "rst_translate_x", 0.0),
            translate_y=getattr(args, "rst_translate_y", 0.0),
            shear=getattr(args, "rst_shear", 0.0),
            resized_crop_ratio=getattr(args, "rst_resized_crop_ratio", None),
            crop_side_ratio=getattr(args, "rst_crop_side_ratio", None),
            order=getattr(args, "rst_order", "affine_then_crop"),
            fill=getattr(args, "rst_fill", 0),
        )

    # Advanced attacks are placed after the standard distortions, matching the
    # common evaluation setting: generate image -> optional geometric/pixel
    # edits -> compression/regeneration attack -> detector.
    attack_order = getattr(args, "advanced_attack_order", "vae_then_regen")
    if attack_order not in ["vae_then_regen", "regen_then_vae"]:
        raise ValueError("advanced_attack_order must be 'vae_then_regen' or 'regen_then_vae'.")

    ordered_names = ["vae", "regen"] if attack_order == "vae_then_regen" else ["regen", "vae"]
    for attack_name in ordered_names:
        attacker = advanced_attackers.get(attack_name, None)
        if attacker is not None:
            img = attacker.attack(img, prompt=prompt)
            img = _ensure_rgb_pil(img)

    return img


def measure_similarity(images, prompt, model, clip_preprocess, tokenizer, device):
    with torch.no_grad():
        img_batch = [clip_preprocess(i).unsqueeze(0) for i in images]
        img_batch = torch.concatenate(img_batch).to(device)
        image_features = model.encode_image(img_batch)

        text = tokenizer([prompt]).to(device)
        text_features = model.encode_text(text)

        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)

        return (image_features @ text_features.T).mean(-1)
