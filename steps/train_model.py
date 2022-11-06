from zenml.steps import step
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.nn as nn
import time

from zenml.steps import step
from torch.utils.data import Dataset
import os
import pandas as pd
import pathlib
from torchvision import transforms
from CustomImageDataset import CustomImageDataset


def get_files_list(directory):
    # get list of files in directory and subdirectories
    files_list = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if not file.endswith(".gif"):
                p = pathlib.Path(os.path.join(root, file))
                files_list.append(pathlib.Path(*p.parts[1:]))
    return files_list


def save_files_list(directory):
    df = pd.DataFrame(get_files_list(directory))
    df.to_csv("../files_list.csv", index=False)


@step()
def train_model(a : list) -> nn.Module:
    model, device = a[0], a[1]
    save_files_list("../cats")
    image_dataset = CustomImageDataset(
        annotations_file="../files_list.csv", img_dir="../cats", transform=transforms.ToTensor())
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    dataloader = DataLoader(image_dataset, batch_size=16, shuffle=True)
    for epoch in range(3):
        start = time.time()
        for data in dataloader:
            img, _ = data
            img = img.to(device)
            # img = img.view(img.size(0), -1)
            optimizer.zero_grad()
            recon = model(img)
            loss = criterion(recon, img)
            loss.backward()
            optimizer.step()

    torch.save(model.state_dict(), "../saved_models/deep_CNN_v3.pth")
