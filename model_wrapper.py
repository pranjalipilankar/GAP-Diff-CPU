# model_wrapper.py
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from models.Generator_Prelayer import Generator_Prelayer
from types import SimpleNamespace

# Load model once (shared across calls)
class GAPDiffModel:
    def __init__(self, weight_path: str, resolution: int = 512, noise_budget: str = "16.0"):
        args = SimpleNamespace(
            generator_path=weight_path,
            resolution=resolution,
            noise_budget=noise_budget,
            training=False  # matches argparse default
        )
        self.device = torch.device("cpu")
        self.model = Generator_Prelayer(args).to(self.device)
        self.model.load_state_dict(torch.load(weight_path, map_location=self.device))
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.Resize((resolution, resolution)),
            transforms.ToTensor(),
            transforms.Normalize([0.5]*3, [0.5]*3)
        ])

    def run(self, img: Image.Image) -> Image.Image:
        img_tensor = self.transform(img.convert("RGB")).unsqueeze(0).to(self.device)

        with torch.no_grad():
            output_tensor = self.model(img_tensor).squeeze(0).cpu()

        # Denormalize
        output_array = output_tensor.permute(1, 2, 0).numpy()
        output_array = (output_array * 0.5 + 0.5) * 255
        output_array = np.clip(output_array, 0, 255).astype(np.uint8)

        return Image.fromarray(output_array)
