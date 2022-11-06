#plik do pracy 
from torchvision.io import read_image
from torchvision.models import resnet50, ResNet50_Weights
import pandas as pd
from PIL import Image
# Step 1: Initialize model with the best available weights
weights = ResNet50_Weights.DEFAULT
model = resnet50(weights=weights)
model.eval()

# Step 2: Initialize the inference transforms
preprocess = weights.transforms()

def is_cat(img):
    #img = read_image(path)

    # Step 3: Apply inference preprocessing transforms
    batch = preprocess(img).unsqueeze(0)

    # Step 4: Use the model and print the predicted category
    prediction = model(batch).squeeze(0).softmax(0)
    px = pd.DataFrame(prediction.detach().numpy())
    px2 = pd.DataFrame(weights.meta["categories"])
    result = pd.concat([px,px2], axis=1)
    result.reset_index(inplace=True)
    result.columns=['index','procent','nazwa']
    result = result[result['index'] >= 281]
    result = result[result['index'] <= 287]
    result = result[result['procent'] >= 0.005]
    return not result.empty
    


def what_is(img):
    #img = read_image(path)
    # Step 3: Apply inference preprocessing transforms
    batch = preprocess(img).unsqueeze(0)

    # Step 4: Use the model and print the predicted category
    prediction = model(batch).squeeze(0).softmax(0)
    px = pd.DataFrame(prediction.detach().numpy())
    px2 = pd.DataFrame(weights.meta["categories"])
    result = pd.concat([px,px2], axis=1)
    result.reset_index(inplace=True)
    result.columns=['index','procent','nazwa']
    result=result.sort_values(by=['procent'],ascending=False)
    return result.head(10)
