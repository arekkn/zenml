from zenml.steps import step
from PIL import Image
import torchvision.models
from utils import eval

resnet34_weights = torchvision.models.ResNet34_Weights.DEFAULT
resnet34_model = torchvision.models.resnet34(weights=resnet34_weights)

@step()
def save_resnet34_embeddings(img_name_list: list) -> list:
    img_embeddings_list = []
    for path in img_name_list:
        img_embeddings_list.append(eval(Image.open(path), resnet34_model, resnet34_weights))
    return img_embeddings_list