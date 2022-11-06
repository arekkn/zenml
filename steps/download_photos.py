from utils import get_cats
from zenml.steps import step


@step()
def download_cat_photos() -> None:
    get_cats(90)

