import os
import time

import numpy as np
import torch
from PIL import Image

# from data.sequence_datasets import indices_TMNIST
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms


class tMNISTDataset(Dataset):
    def __init__(self, data_file, root_dir, transform=None):
        self.labels = np.load(data_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return int(self.labels.size / 2) - 1

    def __getitem__(self, idx):
        [img_name, label] = self.labels[idx + 1]
        image = Image.open(os.path.join(self.root_dir, img_name)).convert("L")

        if self.transform:
            image = self.transform(image)

        return (image, int(label))


if __name__ == "__main__":
    train_dir = "tMNIST_train"

    train_images = None
    train_labels = None
    train_bs = 128
    train_transform = transforms.Compose([transforms.ToTensor()])
    dataset = tMNISTDataset(
        data_file=f"{train_dir}/data.npy", root_dir=train_dir, transform=train_transform
    )
    data_loader = DataLoader(dataset=dataset, batch_size=train_bs, shuffle=True)

    for iteration, (x, y) in enumerate(data_loader):
        if train_images is None:
            train_images, train_labels = x.numpy(), y.numpy()
            continue
        train_images = np.vstack([train_images, x.numpy()])
        train_labels = np.concatenate([train_labels, y.numpy()])
    np.save(os.path.join(train_dir, "train_ds.npy"), train_npy)
