# model_wrapper.py
import os
import uuid
import subprocess
from PIL import Image

class GAPDiffWrapper:
    def __init__(self,
                 generator_path="G_per16_pretrain.pth",
                 resolution=512,
                 noise_budget="16.0"):
        self.generator_path = generator_path
        self.resolution = resolution
        self.noise_budget = noise_budget

    def run(self, img: Image.Image) -> Image.Image:
        # Generate unique ID to avoid filename collisions
        session_id = str(uuid.uuid4())[:8]
        temp_dir = f"temp/{session_id}"
        input_path = os.path.join(temp_dir, "input.png")
        output_path = os.path.join(temp_dir, "output", "input.png")

        # Create necessary directories
        os.makedirs(os.path.join(temp_dir, "output"), exist_ok=True)

        # Save input image
        img.save(input_path)

        # Run generate.py using subprocess
        try:
            subprocess.run([
                "python", "generate.py",
                f"--generator_path={self.generator_path}",
                f"--source_path={temp_dir}",
                f"--save_path={os.path.join(temp_dir, 'output')}",
                f"--resolution={self.resolution}",
                f"--noise_budget={self.noise_budget}"
            ], check=True)
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"generate.py failed: {e}")

        if not os.path.exists(output_path):
            raise FileNotFoundError(f"Output image not found: {output_path}")

        # Load result
        result_img = Image.open(output_path).convert("RGB")

        # (Optional) Cleanup
        # import shutil; shutil.rmtree(temp_dir)

        return result_img
