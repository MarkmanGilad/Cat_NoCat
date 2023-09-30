import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split

from torch.utils.data import DataLoader, TensorDataset
from Class_CNN import CNN_Model


# Device configuration
if torch.cuda.is_available:
  device = torch.device('cuda')
else:
  device = torch.device('cpu')
print(device)

# prepare data
file = 'Data/cats.pth'
data, labels = torch.load(file)

print(data.shape, labels.shape)
# print(data[25])

# Example of a picture
# index1 = 25 # change index to get a different picture
# index2 = 26 # change index to get a different picture
# plt.subplot(1,2,1)
# plt.imshow(data[index1])
# plt.title(labels[index1])
# plt.subplot(1,2,2)
# plt.imshow(data[index2])
# plt.title(labels[index2])
# plt.show()


data_train_np, data_test_np, target_train_np, target_test_np = train_test_split (data,labels, train_size=0.8,random_state=10, shuffle=True)
print ('train np shape: ',data_train_np.shape, target_train_np.shape)
print ('test np shape: ', data_test_np.shape, target_test_np.shape)

data_train = torch.from_numpy(data_train_np.astype(np.float32))
data_train = data_train.permute(0,3,1,2)
target_train = torch.from_numpy(target_train_np.astype(np.float32))
target_train = target_train.view(target_train.shape[0],-1)

data_test = torch.from_numpy(data_test_np.astype(np.float32))
data_test = data_test.permute(0,3,1,2)
target_test = torch.from_numpy(target_test_np.astype(np.float32))
target_test = target_test.view(target_test.shape[0],-1)

print ('train tensor shape: ', data_train.shape, target_train.shape)
print ('test tensor shape: ', data_test.shape, target_test.shape)

# Normalized Data
data_train = data_train / 255
data_test = data_test / 255


# use pytorch dataset and dataloader
batch_size = 25
data_train = TensorDataset(data_train, target_train)
data_test = TensorDataset(data_test, target_test)

train_loader = DataLoader(dataset=data_train, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=data_test, batch_size=batch_size, shuffle=False)

# check sizes of data batches
# for X,y in train_loader:
#   print(X.shape,y.shape)


# get some random training images
dataiter = iter(train_loader)
for i in range(4):
    images, labels = next(dataiter)
    for j in range(batch_size):
        plt.subplot(4,batch_size,i*batch_size+j+1)
        plt.imshow((images[j].permute(1,2,0).numpy()*255).astype(np.int0))
        plt.title(labels[j].item())
plt.show()

# parameters
epochs = 300
learning_rate = 0.001
losses = []



Model = CNN_Model().to(device)

# loss and optimizer
Loss = nn.BCELoss()
optim = torch.optim.Adam(Model.parameters(), lr=learning_rate)

n_total_steps = len(train_loader)


# train loop

for epoch in range(epochs):
    for i, (images, lables) in enumerate(train_loader):

        images = images.to(device)
        lables = lables.to(device)

        # forward
        Y_predict = Model(images)

        # backward
        optim.zero_grad()
        loss = Loss(Y_predict, lables)
        loss.backward()
        losses.append(loss.item())

        if i % 10 == 0:
            print(f"epoch= {epoch} i= {i+epoch * n_total_steps} loss={loss.item():.4f} ")

        # update wights
        optim.step()


# save Model
torch.save(Model,'Data/model.pth')

# print results
plt.plot(losses, 'o')
plt.show()
    
# test Model
with torch.no_grad():
    n_correct = 0
    n_samples = 0
    for images, lables in test_loader:
        images = images.to(device)
        lables = lables.to(device)
        y_predict = Model(images)
        y_predict = y_predict > 0.5
        n_samples += lables.size(0)
        n_correct += (y_predict == lables).sum().item()

    acc = 100 * n_correct / n_samples
    print(f'Accuracy of the network on the {n_samples} test images: {acc} %')

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
