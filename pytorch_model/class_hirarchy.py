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
import os
import torch
import torchvision
import torchvision.transforms as transforms

# define data loader

df = pd.read_csv(r"D:\python\Pytorch_HIC\CIFAR-10_renamed\annotations.csv")


train_set, test_set = train_test_split(df, test_size=0.25)

lineages = [['object', 'driving', 'truck'], ['object', 'driving', 'automobile'],
            ['object', 'not', 'ship'], ['object', 'not', 'airplane'], ['animal', 'mammal', 'deer'],
            ['animal', 'mammal', 'dog'], ['animal', 'mammal', 'cat'], ['animal', 'mammal', 'horse'],
            ['animal', 'other', 'frog'], ['animal', 'other', 'bird']]

batchsize = 4

class ImageDataset(Dataset):
    def __init__(self, csv, image_folder, transform):
        self.csv = csv
        self.transform = transform
        self.folder = folder
        self.image_names = self.csv[:]['id']
        self.lineages = self.csv[:]['lineage']
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



train_dataset = ImageDataset(train_set, folder, transform)
test_dataset = ImageDataset(test_set, folder, transform)

train_dataloader = DataLoader(train_dataset, batch_size=batchsize, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=batchsize, shuffle=True)



# get some random training images
sample = next(iter(train_dataloader))
images = sample['image']
labels = sample['labels']


# define network architecture
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 16)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


net = Net()
# define loss function and optimizer

import torch.optim as optim
# replace with custom loss function
scaling_vector = torch.tensor(np.concatenate((np.array([3, 3, 2, 2, 2, 2]), np.repeat(1, repeats=10))))

def scaled_loss(output, target, scaling_vector):
    loss = torch.mean(((output - target)*scaling_vector)**2)
    return loss


optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

# training loop
for epoch in range(1):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(train_dataloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs = data['image']
        labels = data['labels']
        labels = labels.type(torch.float32)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = scaled_loss(outputs, labels, scaling_vector)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 2000 == 1999:    # print every 2000 mini-batches
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
            running_loss = 0.0

print('Finished Training')



# prediction give error per hirarchy step
outputs = net(images)

correct = 0
total = 0
# since we're not training, we don't need to calculate the gradients for our outputs
with torch.no_grad():
    for data in test_dataloader:
        images, labels = data
        # calculate outputs by running images through the network
        outputs = net(images)
        # setting a threshold for the energy, so we can compare to the labels

        #splitting the outputs into the hirarchy steps and check for each step if the prediction is correct




print(f'Accuracy of the network on the 10000 test images: {100 * correct // total} %')

# prepare to count predictions for each class

# quality metrics