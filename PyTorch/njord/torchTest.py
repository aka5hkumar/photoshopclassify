import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

import matplotlib.pyplot as plt
plt.style.use("ggplot")

# OpenCV Image Library
import cv2

# Import PyTorch
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader, Dataset
import torchvision
import torch.optim as optim

class CreateDataset(Dataset):
    def __init__(self, df_data, data_dir = './', transform=None):
        super().__init__()
        self.df = df_data.values
        self.data_dir = data_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, index):
        img_name,label = self.df[index]
        img_path = os.path.join(self.data_dir, img_name)
        image = cv2.imread(img_path)
        if self.transform is not None:
            image = self.transform(image)
        return image, label


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # Convolutional Layer (sees 32x32x3 image tensor) 
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=4, padding=1)
        # Convolutional Layer (sees 16x16x16 image tensor)
        self.conv2 = nn.Conv2d(64, 128, 4, padding=1)
        # Convolutional Layer (sees 8x8x32 image tensor)
        self.conv3 = nn.Conv2d(128, 256, 4, padding=1)
        # Convolutional Layer (sees 4*4*64 image tensor)
        self.conv4 = nn.Conv2d(256, 512, 4, padding=1)
        # Maxpooling Layer
        self.pool = nn.MaxPool2d(2, 2)
        # Linear Fully-Connected Layer 1 (sees 2*2*128 image tensor)
        self.fc1 = nn.Linear(128*2*2, 512)
        # Linear FC Layer 2
        self.fc2 = nn.Linear(512, 2)
        # Set Dropout
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x):
        # add sequence of convolutional and max pooling layers
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.pool(F.relu(self.conv4(x)))
        # flatten image input
        x = x.view(-1, 128 * 2 * 2)
        # add dropout layer
        x = self.dropout(x)
        # add 1st hidden layer, with relu activation function
        x = F.relu(self.fc1(x))
        # add dropout layer
        x = self.dropout(x)
        # add 2nd hidden layer, with relu activation function
        x = F.relu(self.fc2(x))
        return x
        
# Load Best parameters learned from training into our model to make predictions later
transforms_test = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((32,32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])


# creating test data
batch_size=2
test_path = './data/test_images/'
sample_sub = pd.read_csv("./data/sample_submission.csv")
test_data = CreateDataset(df_data=sample_sub, data_dir=test_path, transform=transforms_test)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

model = CNN()
model.load_state_dict(torch.load('./data/best_model.pt'))

# Turn off gradients
model.eval()


preds = []
for batch_i, (data, target) in enumerate(test_loader):
    #data, target = data.cuda(), target.cuda()
    output = model(data)

    pr = output[:,1].detach().cpu().numpy()
    for i in pr:
        preds.append(i)

# Create Submission file        
sample_sub['has_cactus'] = preds
sample_sub.to_csv('./data/submission.csv', index=False)
