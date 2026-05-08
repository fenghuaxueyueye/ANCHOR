"""Advanced image attacks for ANCHOR watermark evaluation."""

from .vae_compression import VAECompressionAttacker
from .regeneration import DiffusionRegenerationAttacker


def build_advanced_attackers(args, device="cuda"):
    """Build optional advanced attackers once and reuse them for all images.

    Returns a dict. Empty dict means no advanced attack is enabled.
    """
    attackers = {}

    if getattr(args, "vae_attack", False):
        attackers["vae"] = VAECompressionAttacker(
            model_name=getattr(args, "vae_model", "bmshj2018-factorized"),
            quality=getattr(args, "vae_quality", 1),
            device=device,
        )

    if getattr(args, "regen_attack", False):
        attackers["regen"] = DiffusionRegenerationAttacker(
            model_path=getattr(args, "regen_model_path", None) or getattr(args, "model_path", None),
            device=device,
            dtype=getattr(args, "regen_dtype", "float16"),
            noise_step=getattr(args, "regen_noise_step", 60),
            denoise_steps=getattr(args, "regen_denoise_steps", 50),
            guidance_scale=getattr(args, "regen_guidance_scale", 7.5),
            use_prompt=getattr(args, "regen_use_prompt", False),
            seed=getattr(args, "regen_seed", 1024),
            start_step_mode=getattr(args, "regen_start_step_mode", "maxsive"),
        )

    return attackers
