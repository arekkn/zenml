from zenml.steps import step
from PIL import Image
import torchvision.models
from utils import eval
import random
resnet18_weights = torchvision.models.ResNet18_Weights.DEFAULT
resnet18_model = torchvision.models.resnet18(weights=resnet18_weights)

@step()
def save_resnet18_embeddings(img_name_list: list) -> list:
    img_embeddings_list = []
    for path in img_name_list:
        if random.random()<0.05:
            print('dupa')
        img_embeddings_list.append(eval(Image.open(path), resnet18_model, resnet18_weights))
    return img_embeddings_list