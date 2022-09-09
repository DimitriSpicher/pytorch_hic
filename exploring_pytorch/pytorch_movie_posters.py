# https://androidkt.com/load-custom-image-datasets-into-pytorch-dataloader-without-using-imagefolder/
import torch
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import cv2
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import os

df = pd.read_csv(r"D:\python\Pytorch_HIC\Movie_posters\train.csv")
train_set, test_set = train_test_split(df, test_size=0.25)


class ImageDataset(Dataset):
    def __init__(self, csv, folder, transform):
        self.csv = csv
        self.transform = transform
        self.folder = folder
        self.image_names = self.csv[:]['Id']
        self.labels = np.array(self.csv.drop(['Id', 'Genre'], axis=1))

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, index):

        image = cv2.imread(os.path.join(self.folder, f'{self.image_names.iloc[index]}.jpg'))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = self.transform(image)
        targets = self.labels[index]

        sample = {'image': image, 'labels': targets}

        return sample


train_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((200, 200)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=45),
    transforms.ToTensor()])

test_transform =transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((200, 200)),
    transforms.ToTensor()])

folder = r"D:\python\Pytorch_HIC\Movie_posters\image_folder"
batchsize = 4

train_dataset = ImageDataset(train_set, folder, train_transform)
test_dataset = ImageDataset(test_set, folder, test_transform)


train_dataloader = DataLoader(train_dataset, batch_size=batchsize, shuffle=True)

test_dataloader = DataLoader(test_dataset, batch_size=batchsize, shuffle=True)

sample = next(iter(train_dataloader))




