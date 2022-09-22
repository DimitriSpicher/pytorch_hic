import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import cv2
from torch.utils.data import DataLoader, Dataset
import os
import torch
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# this script implements a convolution neural network based on the CIFAR-10 dataset. It is based on a
# standard classification network found here https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
# I have extended the network to handle 3 classes per image which are organised as lineages. For example a truck would
# have the lineage (object-driving-truck)
# I defined a custom loss function to scale the loss for mistakes earlier in the lineage.

# defining the Dataset class which is used to load the data.


class ImageDataset(Dataset):
    def __init__(self, csv, folder, transform):
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

# defining the neural network structure. This is based on an example that was extended to classify 16 classes


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

# custom loss function that scales the loss accordign to the hirarchy of the class


def scaled_loss(output, target, scaling_vector):
    loss = torch.mean(((output - target)*scaling_vector)**2)
    return loss


def convert_label_to_class_index(target):

    # the class hierarchy structure this is hardcoded for now
    class_0_index = target[:, 0:2].argmax(dim=1)
    class_1_index = target[:, 2:6].argmax(dim=1)
    class_2_index = target[:, 6:17].argmax(dim=1)

    return class_0_index, class_1_index, class_2_index


df = pd.read_csv(r"D:\python\Pytorch_HIC\CIFAR-10_renamed\annotations.csv")
folder = r"D:\python\Pytorch_HIC\CIFAR-10_renamed\image_folder"

train_set, test_set = train_test_split(df, test_size=0.25)

# the classes are structured in 3 hierarchy steps. Each image is classified into belonging to one class for each step

class_0 = ['object', 'animal']
class_1 = ['driving', 'not', 'mammal', 'other']
class_2 = ['truck', 'automobile', 'ship', 'airplane', 'deer', 'dog', 'cat', 'horse', 'frog', 'bird']

batchsize = 4

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

train_dataset = ImageDataset(train_set, folder, transform)
test_dataset = ImageDataset(test_set, folder, transform)

train_dataloader = DataLoader(train_dataset, batch_size=batchsize, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=batchsize, shuffle=True)

net = Net()

scaling_vector = torch.tensor(np.concatenate((np.array([3, 3, 2, 2, 2, 2]), np.repeat(1, repeats=10))))

optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

# training loop

for epoch in range(4):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(train_dataloader, 0):

        inputs = data['image']
        labels = data['labels']
        labels = labels.type(torch.float32)

        optimizer.zero_grad()

        outputs = net(inputs)
        loss = scaled_loss(outputs, labels, scaling_vector)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i % 2000 == 1999:    # print every 2000 mini-batches
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
            running_loss = 0.0

print('Finished Training')

# count predictions for each class
correct_class_0 = 0
correct_class_1 = 0
correct_class_2 = 0

total = test_dataset.__len__()

# Counting the correct classification in each hierarchy step

with torch.no_grad():
    for data in test_dataloader:
        labels = data['labels']
        image = data['image']
        outputs = net(image)

        index_0_ground, index_1_ground, index_2_ground = convert_label_to_class_index(labels)

        index_0_predicted, index_1_predicted, index_2_predicted = convert_label_to_class_index(outputs)
        correct_class_0 += torch.sum(index_0_predicted == index_0_ground)
        correct_class_1 += torch.sum(index_1_predicted == index_1_ground)
        correct_class_2 += torch.sum(index_2_predicted == index_2_ground)

print(f'Accuracy of the network for class 0: {100 * correct_class_0/total} %')
print(f'Accuracy of the network for class 1: {100 * correct_class_1/total} %')
print(f'Accuracy of the network for class 2: {100 * correct_class_2/total} %')

# quality metrics comparison to single class network needed
