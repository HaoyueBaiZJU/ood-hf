import numpy as np
import torch
import torch.utils.data as data
import torchvision
from PIL import Image
from random import sample, random


from torchvision import datasets
import torch
from torchvision import transforms
from torch.utils.data import DataLoader, ConcatDataset


import os.path as osp
from torch.utils.data import Dataset
from tqdm import tqdm

import os

from random import sample, random

ROOT_PATH = './datasets/PACS/pacs_data/'

class PACScorDataset(data.Dataset):
    def __init__(self, setname, target_domain):

        self._image_transformer = transforms.Compose([
            transforms.Resize(255, Image.BILINEAR),
        ])
        self._image_transformer_full = transforms.Compose([
            transforms.Resize(225, Image.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        fulldata = []
        label_name = []
        fullconcept = []
        domain = [target_domain]
        for domain_name in domain:
            txt_path = os.path.join(ROOT_PATH, domain_name + '.txt')
            images, labels = self._dataset_info(txt_path)
            fulldata.extend(images)
            label_name.extend(labels)
            concept = [3] * len(labels)
            fullconcept.extend(concept)

        self.data = fulldata
        self.label = label_name
        self.num_class = np.max(self.label) + 1
        self.concept = fullconcept

    def _dataset_info(self, txt_labels):
        with open(txt_labels, 'r') as f:
            images_list = f.readlines()

        file_names = []
        labels = []
        for row in images_list:
            row = row.split(' ')
            path = os.path.join(row[0])
            path = path.replace('\\', '/')

            file_names.append(path)
            labels.append(int(row[1]))

        return file_names, labels

    def get_random_subset(self, names, labels, percent):
        """
        :param names: list of names
        :param labels:  list of labels
        :param percent: 0 < float < 1
        :return:
        """
        samples = len(names)
        amount = int(samples * percent)
        random_index = sample(range(samples), amount)
        name_val = [names[k] for k in random_index]
        name_train = [v for k, v in enumerate(names) if k not in random_index]
        labels_val = [labels[k] for k in random_index]
        labels_train = [v for k, v in enumerate(labels) if k not in random_index]

        return name_train, name_val, labels_train, labels_val
    
    def __getitem__(self, index):
        data, label, concept= self.data[index], self.label[index], self.concept[index]

        _img = Image.open(data).convert('RGB')
        img = self._image_transformer_full(_img)

        return img, label

    def __len__(self):
        return len(self.data)

