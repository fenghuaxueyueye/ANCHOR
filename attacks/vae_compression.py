"""VAE / neural image compression attack for ANCHOR.

This follows the same practical attack family used in MaXsive/GaussMarker-style
benchmarks: compress the generated image with a learned CompressAI model and
feed the reconstructed image into the watermark detector.
"""

from PIL import Image
import torch
from torchvision import transforms


class VAECompressionAttacker:
    """CompressAI learned-compression attack.

    Supported models match CompressAI zoo names commonly used in watermarking
    benchmarks:
      - bmshj2018-factorized
      - bmshj2018-hyperprior
      - mbt2018-mean
      - mbt2018
      - cheng2020-anchor

    quality is the CompressAI quality level. For most models valid levels are
    1--8; lower values usually mean stronger compression.
    """

    def __init__(self, model_name="bmshj2018-factorized", quality=1, device="cuda"):
        self.model_name = model_name
        self.quality = int(quality)
        self.device = device

        try:
            from compressai.zoo import (
                bmshj2018_factorized,
                bmshj2018_hyperprior,
                mbt2018_mean,
                mbt2018,
                cheng2020_anchor,
            )
        except Exception as exc:
            raise ImportError(
                "VAE compression attack requires CompressAI. Install it with: "
                "pip install compressai"
            ) from exc

        builders = {
            "bmshj2018-factorized": bmshj2018_factorized,
            "bmshj2018-hyperprior": bmshj2018_hyperprior,
            "mbt2018-mean": mbt2018_mean,
            "mbt2018": mbt2018,
            "cheng2020-anchor": cheng2020_anchor,
        }
        if model_name not in builders:
            raise ValueError(
                f"Unsupported VAE compression model '{model_name}'. "
                f"Choose one of {list(builders.keys())}."
            )

        self.model = builders[model_name](quality=self.quality, pretrained=True)
        self.model = self.model.eval().to(device)

    @torch.no_grad()
    def attack(self, image: Image.Image, prompt: str = "") -> Image.Image:
        if not isinstance(image, Image.Image):
            raise TypeError("VAECompressionAttacker expects a PIL.Image input.")

        original_size = image.size
        img = image.convert("RGB")
        tensor = transforms.ToTensor()(img).unsqueeze(0).to(self.device)

        out = self.model(tensor)
        x_hat = out["x_hat"].clamp(0, 1).squeeze(0).cpu()
        attacked = transforms.ToPILImage()(x_hat).convert("RGB")

        # Keep the detector input resolution consistent with the original image.
        if attacked.size != original_size:
            attacked = attacked.resize(original_size, Image.BICUBIC)
        return attacked
