import os
import time
from random import randint

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image
from tqdm import tqdm

# Number of images to generate
sample_count_train = 60000
sample_count_val = 10000

# Batch size (no. samples per subfolder)
batch_size = 128

# Image size of the generated image, higher number means a greater transformation
WIDTH = 40
HEIGHT = 40

# Image size of the dataset, MNIST is 28x28
data_x = 28
data_y = 28

train_dir = "tMNIST_train"
val_dir = "tMNIST_val"

assert data_x < WIDTH
assert data_y < HEIGHT

try:
    os.mkdir(dir_name)
except FileExistsError:
    print("Directory %s already exists." % dir_name)

# Initialize dataframe for labels
data = np.array(["filename", "label"])

dataset = datasets.MNIST(
    root="data", train=True, transform=transforms.ToTensor(), download=True
)
data_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False)

current_index = 0

with tqdm(total=sample_count_train) as pbar:
    for iteration, (x, y) in enumerate(data_loader):
        if current_index == sample_count_train:
            break

        current_batch_size = x.size(0)

        batch_dir_name = "batch_%d" % iteration

        try:
            os.mkdir(os.path.join(train_dir, batch_dir_name))
        except FileExistsError:
            pass

        for i in range(current_batch_size):
            base_image = np.zeros((HEIGHT, WIDTH))

            rand_x = randint(0, WIDTH - (data_x + 1))
            rand_y = randint(0, HEIGHT - (data_y + 1))

            base_image[rand_y : rand_y + data_y, rand_x : rand_x + data_x] = (
                x.detach().numpy()[i].reshape(data_y, data_x)
            )

            filename = "{0:010d}.png".format(current_index)
            save_image(
                torch.Tensor(base_image),
                os.path.join(train_dir, batch_dir_name, filename),
            )

            data = np.vstack(
                [data, [os.path.join(batch_dir_name, filename), y[i].item()]]
            )

            pbar.update(1)
            current_index += 1

            if current_index == sample_count_train:
                break

    np.save(os.path.join(train_dir, "data.npy"), data)
pbar.close()

with tqdm(total=sample_count_val) as pbar:
    for iteration, (x, y) in enumerate(data_loader):
        if current_index == sample_count_val:
            break

        current_batch_size = x.size(0)

        batch_dir_name = "batch_%d" % iteration

        try:
            os.mkdir(os.path.join(val_dir, batch_dir_name))
        except FileExistsError:
            pass

        for i in range(current_batch_size):
            base_image = np.zeros((HEIGHT, WIDTH))

            rand_x = randint(0, WIDTH - (data_x + 1))
            rand_y = randint(0, HEIGHT - (data_y + 1))

            base_image[rand_y : rand_y + data_y, rand_x : rand_x + data_x] = (
                x.detach().numpy()[i].reshape(data_y, data_x)
            )

            filename = "{0:010d}.png".format(current_index)
            save_image(
                torch.Tensor(base_image),
                os.path.join(val_dir, batch_dir_name, filename),
            )

            data = np.vstack(
                [data, [os.path.join(batch_dir_name, filename), y[i].item()]]
            )

            pbar.update(1)
            current_index += 1

            if current_index == sample_count_val:
                break

    np.save(os.path.join(val_dir, "data.npy"), data)
pbar.close()
