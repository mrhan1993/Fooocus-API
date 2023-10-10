# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md

from cog import BasePredictor, Input, Path

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
        image: Path = Input(description="Grayscale input image"),
        scale: float = Input(
            description="Factor to scale image by", ge=0, le=10, default=1.5
        ),
    ) -> Path:
        """Run a single prediction on the model"""
        print("Predictor predict")
        # processed_input = preprocess(image)
        # output = self.model(processed_image, scale)
        # return postprocess(output)
