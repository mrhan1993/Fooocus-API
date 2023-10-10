# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md

import os
from typing import List
from cog import BasePredictor, Input, Path

from fooocusapi.models import GenerationFinishReason, Text2ImgRequest
from fooocusapi.worker import process_generate
from modules.util import generate_temp_filename
from PIL import Image


class Args(object):
    sync_repo = None


class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""
        from main import prepare_environments
        prepare_environments(Args())

        import modules.default_pipeline as _
        print("Predictor setuped")

    def predict(
        self,
        prompt: str = Input(
            default='', description="Prompt for image generation")
    ) -> List[Path]:
        """Run a single prediction on the model"""
        text_to_img_req = Text2ImgRequest(prompt=prompt)
        results = process_generate(text_to_img_req)

        output_paths: List[Path] = []
        for r in results:
            if r.finish_reason == GenerationFinishReason.success and r.im is not None:
                output_path = generate_temp_filename('/tmp')
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                Image.fromarray(r.im).save(output_path)
                output_paths.append(Path(output_path))

        return output_paths
