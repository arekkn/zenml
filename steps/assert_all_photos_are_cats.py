from is_cat import is_cat
from zenml.steps import step
from PIL import Image
import os


@step()
def check_cats() -> None:

    directory = '../cats'
    for filename in os.listdir(directory):
        f = os.path.join(directory, filename)
        image = Image.open(f).convert('RGB')
        if not is_cat(image):
            os.remove(f)
