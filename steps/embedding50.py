from zenml.steps import step
from PIL import Image
import torchvision.models
from utils import eval

resnet50_weights = torchvision.models.ResNet50_Weights.DEFAULT
resnet50_model = torchvision.models.resnet50(weights=resnet50_weights)

@step()
def save_resnet50_embeddings(img_name_list: list) -> list:
    img_embeddings_list = []
    for path in img_name_list:
        img_embeddings_list.append(eval(Image.open(path), resnet50_model, resnet50_weights))
    return img_embeddings_list