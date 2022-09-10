# thoughts
# the class is represented as a tree. ech instance has a certain lineage in that tree
# the tree gets flattend into a vector to calculate loss. Multipl loss vector with a scaling vector
# to make earlier nodes more important maybe not needed.

# import stuff
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import cv2
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import os
import torchvision

# define data loader

df = pd.read_csv(r"D:\python\Pytorch_HIC\CIFAR-10_renamed\annotations.csv")

train_set, test_set = train_test_split(df, test_size=0.25)



class ImageDataset(Dataset):
    def __init__(self, csv, image_folder, transform):
        self.csv = csv
        self.transform = transform
        self.folder = folder
        self.image_names = self.csv[:]['id']
        self.labels = np.array(self.csv.drop(['id', 'lineage'], axis=1))

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, index):

        image = cv2.imread(os.path.join(self.folder, self.image_names.iloc[index]))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = self.transform(image)
        targets = self.labels[index]

        sample = {'image': image, 'labels': targets}

        return sample


transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

folder = r"D:\python\Pytorch_HIC\CIFAR-10_renamed\image_folder"

batchsize = 4

train_dataset = ImageDataset(train_set, folder, transform)
test_dataset = ImageDataset(test_set, folder, transform)

train_dataloader = DataLoader(train_dataset, batch_size=batchsize, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=batchsize, shuffle=True)


import matplotlib.pyplot as plt
import numpy as np

def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


# get some random training images
dataiter = iter(train_dataloader)
images, labels = next(dataiter)

# show images
imshow(torchvision.utils.make_grid(images))



# print labels
print(' '.join(f'{classes[labels[j]]:5s}' for j in range(batch_size)))
# define network architecture

# define loss function and optimizer

# training loop

# prediction

# quality metrics