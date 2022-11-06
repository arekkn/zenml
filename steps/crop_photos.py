from utils import crop_and_scale
from zenml.steps import step
import os
from PIL import Image


@step()
def format_photos() -> None:
    directory = '../cats'
    for filename in os.listdir(directory):
        f = os.path.join(directory, filename)
        if not f.endswith('gif'):
            img = Image.open(f).convert('RGB')
            crop_and_scale(img).save(f)
        else:
            os.remove(f)

