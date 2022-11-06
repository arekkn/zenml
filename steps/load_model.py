from zenml.steps import step
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import time
import numpy as np
from PIL import Image
from Autoencoder import Autoencoder



@step()
def load_model() -> list :
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = Autoencoder()
    loaded_model = torch.load("../saved_models/deep_CNN_v3.pth")
    model.load_state_dict(loaded_model)
    model.eval()
    model.to(device)
    return [model, device]
