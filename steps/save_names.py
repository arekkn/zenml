from zenml.steps import step
import os

@step(enable_cache=False)
def save_names() -> list:
    directory="../cats/"
    img_name_list = []
    for filename in os.listdir(directory):
        f = os.path.join(directory, filename)
        img_name_list.append(f)
    return img_name_list