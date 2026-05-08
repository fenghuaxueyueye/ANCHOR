"""MaXsive-style diffusion regeneration attack for ANCHOR.

This version does NOT use the img2img ``strength`` parameter.  It follows the
logic used in MaXsive more closely:

    PIL image -> VAE encode -> add Gaussian noise at a diffusion timestep
    ``noise_step`` -> denoise/regenerate from the corresponding late denoising
    step -> PIL image.

The attack strength is controlled mainly by ``noise_step``.  A larger
``noise_step`` means more noise is injected into the image latent, so the
regenerated image is more strongly changed and the watermark is more strongly
attacked.
"""

from PIL import Image
import numpy as np
import torch
from torchvision import transforms


class DiffusionRegenerationAttacker:
    """Diffusion regeneration attack controlled by noise timestep.

    Parameters
    ----------
    model_path:
        Local or HF path of the Stable Diffusion model.
    noise_step:
        Diffusion training timestep used to add noise to the encoded image
        latent. This matches the MaXsive-style regeneration attack parameter.
        Typical values: 20, 40, 60, 80, 100. Larger means stronger attack.
    denoise_steps:
        Number of denoising steps used by the regeneration pipeline. Usually 50.
    guidance_scale:
        Classifier-free guidance scale used in regeneration. MaXsive uses 7.5.
    use_prompt:
        If False, use an empty prompt for regeneration. If True, use the image's
        original generation prompt passed to attack(..., prompt=...).
    start_step_mode:
        "maxsive": use denoise_steps - max(noise_step // 20, 1), following
        the public MaXsive implementation.
        "nearest": choose the denoising index whose scheduler timestep is
        closest to noise_step.
    """

    def __init__(
        self,
        model_path,
        device="cuda",
        dtype="float16",
        noise_step=60,
        denoise_steps=50,
        guidance_scale=7.5,
        use_prompt=False,
        seed=1024,
        start_step_mode="maxsive",
    ):
        if model_path is None:
            raise ValueError("model_path is required for diffusion regeneration attack.")

        try:
            from diffusers import StableDiffusionPipeline, DDIMScheduler
        except Exception as exc:
            raise ImportError("Diffusion regeneration attack requires diffusers.") from exc

        self.model_path = model_path
        self.device = device
        self.noise_step = int(noise_step)
        self.denoise_steps = int(denoise_steps)
        self.guidance_scale = float(guidance_scale)
        self.use_prompt = bool(use_prompt)
        self.seed = int(seed)
        self.start_step_mode = start_step_mode

        if self.denoise_steps <= 0:
            raise ValueError("regen_denoise_steps must be positive.")
        if self.noise_step < 0:
            raise ValueError("regen_noise_step must be non-negative.")
        if self.start_step_mode not in ["maxsive", "nearest"]:
            raise ValueError("regen_start_step_mode must be 'maxsive' or 'nearest'.")

        torch_dtype = torch.float16 if dtype == "float16" else torch.float32
        self.pipe = StableDiffusionPipeline.from_pretrained(
            model_path,
            torch_dtype=torch_dtype,
            safety_checker=None,
        )
        try:
            self.pipe.scheduler = DDIMScheduler.from_config(self.pipe.scheduler.config)
        except Exception:
            pass
        self.pipe.set_progress_bar_config(disable=True)
        self.pipe = self.pipe.to(device)
        self.pipe.safety_checker = None

        # Clamp noise_step to scheduler training range if available.
        max_train_t = getattr(self.pipe.scheduler.config, "num_train_timesteps", 1000) - 1
        if self.noise_step > max_train_t:
            self.noise_step = int(max_train_t)

        # Diffusers versions differ: newer VAE configs expose
        # config.scaling_factor, older FrozenDict configs require
        # dictionary-style access, and very old SD configs may not store it.
        self.vae_scaling_factor = self._get_vae_scaling_factor()

    def _get_vae_scaling_factor(self):
        config = self.pipe.vae.config
        if hasattr(config, "scaling_factor"):
            return float(config.scaling_factor)
        if isinstance(config, dict) and "scaling_factor" in config:
            return float(config["scaling_factor"])
        if hasattr(config, "get"):
            value = config.get("scaling_factor", None)
            if value is not None:
                return float(value)
        # Stable Diffusion VAE default.
        return 0.18215

    def _encode_prompt(self, prompt):
        """Encode prompt with compatibility across diffusers versions."""
        do_cfg = self.guidance_scale > 1.0
        device = self.device

        if hasattr(self.pipe, "_encode_prompt"):
            return self.pipe._encode_prompt(
                prompt,
                device,
                num_images_per_prompt=1,
                do_classifier_free_guidance=do_cfg,
                negative_prompt=None,
            )

        # Newer diffusers may expose encode_prompt instead.
        prompt_embeds = self.pipe.encode_prompt(
            prompt,
            device=device,
            num_images_per_prompt=1,
            do_classifier_free_guidance=do_cfg,
            negative_prompt=None,
        )
        if isinstance(prompt_embeds, tuple):
            if do_cfg:
                return torch.cat([prompt_embeds[1], prompt_embeds[0]], dim=0)
            return prompt_embeds[0]
        return prompt_embeds

    def _pil_to_latent(self, image, generator):
        image = image.convert("RGB")
        tensor = transforms.ToTensor()(image).unsqueeze(0).to(self.device)
        tensor = tensor * 2.0 - 1.0
        tensor = tensor.to(dtype=self.pipe.vae.dtype)
        latent_dist = self.pipe.vae.encode(tensor).latent_dist
        latents = latent_dist.sample(generator=generator)
        latents = latents * self.vae_scaling_factor
        return latents

    def _decode_latent(self, latents, original_size):
        latents = latents / self.vae_scaling_factor
        image = self.pipe.vae.decode(latents).sample
        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.detach().cpu().permute(0, 2, 3, 1).float().numpy()[0]
        image = (image * 255).round().astype(np.uint8)
        image = Image.fromarray(image).convert("RGB")
        if image.size != original_size:
            image = image.resize(original_size, Image.BICUBIC)
        return image

    def _get_head_start_step(self):
        if self.start_step_mode == "maxsive":
            # This reproduces the public MaXsive relation:
            # head_start_step = 50 - max(noise_step // 20, 1)
            step = self.denoise_steps - max(self.noise_step // 20, 1)
            return max(0, min(self.denoise_steps - 1, int(step)))

        # Alternative: start from the scheduler timestep closest to noise_step.
        timesteps = self.pipe.scheduler.timesteps
        target = torch.tensor(float(self.noise_step), device=timesteps.device)
        idx = int(torch.argmin(torch.abs(timesteps.float() - target)).item())
        return max(0, min(len(timesteps) - 1, idx))

    @torch.no_grad()
    def attack(self, image: Image.Image, prompt: str = "") -> Image.Image:
        if not isinstance(image, Image.Image):
            raise TypeError("DiffusionRegenerationAttacker expects a PIL.Image input.")

        original_size = image.size
        prompt_used = prompt if self.use_prompt else ""
        generator = torch.Generator(device=self.device).manual_seed(self.seed)

        # 1. Encode attacked image into SD latent space.
        latents = self._pil_to_latent(image, generator)

        # 2. Add Gaussian noise at the selected diffusion timestep.
        timestep = torch.tensor([self.noise_step], dtype=torch.long, device=self.device)
        # torch.randn_like(..., generator=...) is not available in some torch versions.
        noise = torch.randn(
            latents.shape,
            generator=generator,
            device=latents.device,
            dtype=latents.dtype,
        )
        latents = self.pipe.scheduler.add_noise(latents, noise, timestep).to(dtype=self.pipe.unet.dtype)

        # 3. Denoise/regenerate from the corresponding late denoising stage.
        self.pipe.scheduler.set_timesteps(self.denoise_steps, device=self.device)
        timesteps = self.pipe.scheduler.timesteps
        head_start_step = self._get_head_start_step()
        text_embeddings = self._encode_prompt(prompt_used)
        do_cfg = self.guidance_scale > 1.0

        extra_step_kwargs = {}
        if hasattr(self.pipe, "prepare_extra_step_kwargs"):
            try:
                extra_step_kwargs = self.pipe.prepare_extra_step_kwargs(generator, eta=0.0)
            except TypeError:
                extra_step_kwargs = self.pipe.prepare_extra_step_kwargs(generator, 0.0)

        for i, t in enumerate(timesteps):
            if i < head_start_step:
                continue

            latent_model_input = torch.cat([latents] * 2) if do_cfg else latents
            latent_model_input = self.pipe.scheduler.scale_model_input(latent_model_input, t)

            noise_pred = self.pipe.unet(
                latent_model_input,
                t,
                encoder_hidden_states=text_embeddings,
            ).sample

            if do_cfg:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)

            latents = self.pipe.scheduler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample

        # 4. Decode back to image space.
        return self._decode_latent(latents, original_size)
