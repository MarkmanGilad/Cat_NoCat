import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split

from torch.utils.data import DataLoader, TensorDataset

# Device configuration
if torch.cuda.is_available:
  device = torch.device('cuda')
else:
  device = torch.device('cpu')
print(device)

class CNN_Model(nn.Module):
    def __init__(self):
        super(CNN_Model, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 13 * 13, 120)
        self.fc2 = nn.Linear(120, 60)
        self.fc3 = nn.Linear(60, 1)

    def forward(self, x):
        # -> n, 3, 64, 64
        x = self.pool(F.relu(self.conv1(x)))  # -> n, 6, 30, 30
        x = self.pool(F.relu(self.conv2(x)))  # -> n, 16, 13, 13
        x = x.view(-1, 16 * 13 * 13)            # -> n, 2704
        x = F.relu(self.fc1(x))               # -> n, 120
        x = F.relu(self.fc2(x))               # -> n, 84
        x = self.fc3(x)                       # -> n, 10
        x = torch.sigmoid(x)
        return x


# prepare data
file = 'Data/cats.pth'
data, labels = torch.load(file)

print(data.shape, labels.shape)


data_train_np, data_test_np, target_train_np, target_test_np = train_test_split (data,labels, train_size=0.8,random_state=10, shuffle=True)
data_train = torch.from_numpy(data_train_np.astype(np.float32))
data_train = data_train.permute(0,3,1,2)
target_train = torch.from_numpy(target_train_np.astype(np.float32))
target_train = target_train.view(target_train.shape[0],-1)

data_test = torch.from_numpy(data_test_np.astype(np.float32))
data_test = data_test.permute(0,3,1,2)
target_test = torch.from_numpy(target_test_np.astype(np.float32))
target_test = target_test.view(target_test.shape[0],-1)


# Normalized Data
data_train = data_train / 255
data_test = data_test / 255


# use pytorch dataset and dataloader
batch_size = 25
data_train = TensorDataset(data_train, target_train)
data_test = TensorDataset(data_test, target_test)

train_loader = DataLoader(dataset=data_train, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=data_test, batch_size=batch_size, shuffle=False)

Model = torch.load('Data/model.pth')
Model.eval()

with torch.no_grad():
    dataiter = iter(train_loader)
    for i in range(1):
        images, labels = next(dataiter)
        images_gpu = images.to(device)
        y_predict = Model(images_gpu).cpu()
        y_predict = y_predict > 0.5
        for j in range(batch_size):
            plt.subplot(4,batch_size,i*batch_size+j+1)
            plt.imshow((images[j].permute(1,2,0).numpy()*255).astype(np.int0))
            plt.title(y_predict[j].item())
    plt.show()