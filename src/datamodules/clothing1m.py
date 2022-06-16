import os
import argparse
import glob
import numpy as np
from PIL import Image
import torch
from torch.utils.data.sampler import Sampler
from torch.utils.data import random_split, DataLoader, Dataset, Subset
import torchvision.transforms as transforms
import pdb

class Clothing1M(Dataset):
    r"""https://github.com/LiJunnan1992/MLNT"""

    def __init__(self, root, mode, transform=None, percent_clean=None):
        self.root = root
        self.anno_dir = os.path.join(self.root, "annotations")
        self.transform = transform
        self.mode = mode
        self.percent_clean = percent_clean

        self.imgs = []
        self.labels = {}
        
        if self.mode == "dirty_train":
            img_list_file = "clean_train_key_list.txt"
            label_list_file = "clean_label_kv.txt"
            self.img_paths(img_list_file)
            self.gen_labels(label_list_file)
            img_list_file = "clean_val_key_list.txt"
            label_list_file = "clean_label_kv.txt"
            self.img_paths(img_list_file)
            self.gen_labels(label_list_file)
            
            self.clean_count = len(self.imgs)
            if percent_clean is not None:
                self.number_noisy = int(self.clean_count * (100-percent_clean)/percent_clean)
            img_list_file = "noisy_train_key_list.txt"
            label_list_file = "noisy_label_kv.txt"
            self.img_paths(img_list_file)
            self.gen_labels(label_list_file)
        
        if self.mode == "noisy_train":
            img_list_file = "noisy_train_key_list.txt"
            label_list_file = "noisy_label_kv.txt"
            self.img_paths(img_list_file)
            self.gen_labels(label_list_file)

        if self.mode == "train":
            img_list_file = "clean_train_key_list.txt"
            label_list_file = "clean_label_kv.txt"
            self.img_paths(img_list_file)
            self.gen_labels(label_list_file)
            self.clean_count = len(self.imgs)
            if percent_clean is not None:
                self.number_noisy = int(self.clean_count * (100-percent_clean)/percent_clean)
            img_list_file = "noisy_train_key_list.txt"
            label_list_file = "noisy_label_kv.txt"
            self.img_paths(img_list_file)
            self.gen_labels(label_list_file)
            
            if percent_clean is not None:
                state = np.random.get_state()
                np.random.seed(0)
                noisy_selection = np.random.choice(np.arange(self.clean_count, len(self.imgs)), size=self.number_noisy, replace=False)
                np.random.set_state(state)
                self.imgs = np.append(np.array(self.imgs[:self.clean_count]), np.array(self.imgs)[noisy_selection])
                self.clean_indicator = np.zeros(len(self.imgs))
                self.clean_indicator[:self.clean_count] = 1

        if self.mode == "val":
            # img_list_file = "clean_train_key_list.txt"
            # label_list_file = "clean_label_kv.txt"
            # self.img_paths(img_list_file)
            # self.gen_labels(label_list_file)
            
            img_list_file = "clean_val_key_list.txt"
            label_list_file = "clean_label_kv.txt"
            self.img_paths(img_list_file)
            self.gen_labels(label_list_file)
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
        self.sequence = np.arange(len(self.imgs))

    def indicate_clean(self, indices):
        if self.percent_clean is not None:
            return self.clean_indicator[indices.cpu().numpy()]
        else:
            return 0

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
        idx = self.sequence[index]
        img_path = self.imgs[idx]
        target = self.labels[img_path]

        image = Image.open(img_path).convert("RGB")
        img = self.transform(image)
        return idx, img, target

    def __len__(self):
        return len(self.sequence)