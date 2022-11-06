from zenml.steps import step
from PIL import Image
import torchvision.models
from utils import eval

resnet152_weights = torchvision.models.ResNet152_Weights.DEFAULT
resnet152_model = torchvision.models.resnet152(weights=resnet152_weights)

@step()
def save_resnet152_embeddings(img_name_list: list) -> list:
    img_embeddings_list = []
    for path in img_name_list:
        img_embeddings_list.append(eval(Image.open(path), resnet152_model, resnet152_weights))
    return img_embeddings_list