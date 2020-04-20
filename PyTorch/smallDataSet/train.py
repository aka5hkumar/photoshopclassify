import os
import cv2
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
#based on https://pythonprogramming.net/gpu-deep-learning-neural-network-pytorch/

class photoShopped():
    img_Size = 1000
    Photoshop = "./data/photoShopped"
    Original = "./data/original"
    test_Set = "./data/test"
    LABELS = {Photoshop: 0, Original: 1}
    training_data = []


    def labelData(self):
        for label in self.LABELS:
            print(label)
            for image in tqdm(os.listdir(label)):
                if "jpg" in image:
                    try:
                        path = os.path.join(label, image)
                        scaledImage = cv2.imread(path, cv2.imread)
                        scaledImage = cv2.resize(scaledImage, (self.img_Size, self.img_Size))
                        self.training_data.append([np.array(scaledImage), np.eye(2)[self.LABELS[label]]]) 
                        
                    except Exception as e:
                        print (str(e))

        np.random.shuffle(self.training_data)
        np.save("training_data.npy", self.training_data)


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1,32,5)
        self.conv2 = nn.Conv2d(32,64,5)
        self.conv3 = nn.Conv2d(64,128,5)

        x=torch.randn(50,50).view(-1,1,50,50) #Does this need to match the image size?
        self._to_linear = None
        self.convs(x)

        self.fc1 = nn.Linear(self._to_linear,512)
        self.fc2 = nn.Linear(512,2)

    def convs(self, x):
        # max pooling over 2x2
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv3(x)), (2, 2))

        if self._to_linear is None:
            self._to_linear = x[0].shape[0]*x[0].shape[1]*x[0].shape[2]
        return x

    def forward(self, x):
        x = self.convs(x)
        x = x.view(-1, self._to_linear)  
        x = F.relu(self.fc1(x))
        x = self.fc2(x) 
        return F.softmax(x, dim=1)

net = Net()
print(net)

#call on API if not local, blah blah blah. 
training_data = np.load("training_data.npy", allow_pickle=True)
print(len(training_data))

#Why is none of the below in a function and what does it doooooo (so i can name the variables better :)
#might be the cpu run
# optimizer = optim.Adam(net.parameters(), lr=0.001)
# loss_function = nn.MSELoss()

# X = torch.Tensor([i[0] for i in training_data]).view(-1,50,50)
# X = X/255.0
# y = torch.Tensor([i[1] for i in training_data])

# VAL_PCT = 0.1  # lets reserve 10% of our data for validation
# val_size = int(len(X)*VAL_PCT)

# train_X = X[:-val_size]
# train_y = y[:-val_size]

# test_X = X[-val_size:]
# test_y = y[-val_size:]

# BATCH_SIZE = 100
# EPOCHS = 1
EPOCHS = 3
def train(net):
    optimizer = optim.Adam(net.parameters(), lr=0.001)
    BATCH_SIZE = 100
    EPOCHS = 3
    for epoch in range(EPOCHS):
        for i in range(0, len(train_X), BATCH_SIZE): # from 0, to the len of x, stepping BATCH_SIZE at a time. [:50] ..for now just to dev
            #print(f"{i}:{i+BATCH_SIZE}")
            batch_X = train_X[i:i+BATCH_SIZE].view(-1, 1, 50, 50)
            batch_y = train_y[i:i+BATCH_SIZE]

            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            net.zero_grad()

            optimizer.zero_grad()   # zero the gradient buffers
            outputs = net(batch_X)
            loss = loss_function(outputs, batch_y)
            loss.backward()
            optimizer.step()    # Does the update

        print(f"Epoch: {epoch}. Loss: {loss}")

def test(net):
    correct = 0
    total = 0
    with torch.no_grad():
        for i in tqdm(range(len(test_X))):
            real_class = torch.argmax(test_y[i]).to(device)
            net_out = net(test_X[i].view(-1, 1, 50, 50).to(device))[0]  # returns a list, 
            predicted_class = torch.argmax(net_out)

            if predicted_class == real_class:
                correct += 1
            total += 1

    print("Accuracy: ", round(correct/total, 3))

    if __name__ == "__main__":
        if torch.cuda.is_available():
            device = torch.device("cuda:0")  # you can continue going on here, like cuda:1 cuda:2....etc. 
            print("Running on the GPU")
        else:
            device = torch.device("cpu")
            print("Running on the CPU")
        net.to(device)
        train(net)
        test(net)