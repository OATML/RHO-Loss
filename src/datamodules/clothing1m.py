import os
import argparse
import glob
import numpy as np
from PIL import Image
import torch
from torch.utils.data.sampler import Sampler
from torch.utils.data import random_split, DataLoader, Dataset, Subset
import torchvision.transforms as transforms

class Clothing1M(Dataset):
    r"""https://github.com/LiJunnan1992/MLNT"""

    def __init__(self, root, mode, transform):
        self.root = root
        self.anno_dir = os.path.join(self.root, "annotations")
        self.transform = transform
        self.mode = mode

        self.imgs = []
        self.labels = {}

        if self.mode == "train":
            img_list_file = "noisy_train_key_list.txt"
            label_list_file = "noisy_label_kv.txt"
            self.img_paths(img_list_file)
            self.gen_labels(label_list_file)

            img_list_file = "clean_train_key_list.txt"
            label_list_file = "clean_label_kv.txt"
            self.img_paths(img_list_file)
            self.gen_labels(label_list_file)
        if self.mode == "val":
            img_list_file = "clean_train_key_list.txt"
            label_list_file = "clean_label_kv.txt"
            self.img_paths(img_list_file)
            self.gen_labels(label_list_file)
            
            # img_list_file = "clean_val_key_list.txt"
            # label_list_file = "clean_label_kv.txt"
            # self.img_paths(img_list_file)
            # self.gen_labels(label_list_file)
        elif self.mode == "test":
            img_list_file = "clean_test_key_list.txt"
            label_list_file = "clean_label_kv.txt"
            self.img_paths(img_list_file)
            self.gen_labels(label_list_file)

        self.classes = [
            "T-Shirt",
            "Shirt",
            "Knitwear",
            "Chiffon",
            "Sweater",
            "Hoodie",
            "Windbreaker",
            "Jacket",
            "Downcoat",
            "Suit",
            "Shawl",
            "Dress",
            "Vest",
            "Underwear",
        ]
    
    def img_paths(self, img_list_file):
        with open(os.path.join(self.anno_dir, img_list_file), "r") as f:
            lines = f.read().splitlines()
            for l in lines:
                self.imgs.append(os.path.join(self.root, l))
    
    def gen_labels(self, label_list_file):
        with open(os.path.join(self.anno_dir, label_list_file), "r") as f:
            lines = f.read().splitlines()
            for l in lines:
                entry = l.split()
                img_path = os.path.join(self.root, entry[0])
                self.labels[img_path] = int(entry[1])

    def __getitem__(self, index):
        img_path = self.imgs[index]
        target = self.labels[img_path]

        image = Image.open(img_path).convert("RGB")
        img = self.transform(image)
        return index, img, target

    def __len__(self):
        return len(self.imgs)