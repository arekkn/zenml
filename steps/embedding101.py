from zenml.steps import step
from PIL import Image
import torchvision.models
from utils import eval
import random
resnet101_weights = torchvision.models.ResNet101_Weights.DEFAULT
resnet101_model = torchvision.models.resnet101(weights=resnet101_weights)

@step()
def save_resnet101_embeddings(img_name_list: list) -> list:
    img_embeddings_list = []
    for path in img_name_list:
        img_embeddings_list.append(eval(Image.open(path), resnet101_model, resnet101_weights))
    return img_embeddings_list
