import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
import torch.nn.init as init

# prepare data

file = 'Data/cats.pth'
data, labels = torch.load(file)

data_train_np, data_test_np, label_train_np, label_test_np = train_test_split (data,labels, test_size=0.2,random_state=10)
print ('train np shape: ',data_train_np.shape, label_train_np.shape)
print ('test np shape: ', data_test_np.shape, label_test_np.shape)

data_train = torch.from_numpy(data_train_np.astype(np.float32))
data_train = data_train.view(data_train.shape[0],-1)
label_train = torch.from_numpy(label_train_np.astype(np.float32))
label_train = label_train.view(label_train.shape[0],-1)

data_test = torch.from_numpy(data_test_np.astype(np.float32))
data_test = data_test.view(data_test.shape[0], -1)
label_test = torch.from_numpy(label_test_np.astype(np.float32))
label_test = label_test.view(label_test.shape[0],-1)

print ('train tensor shape: ', data_train.shape, label_train.shape)
print ('test tensor shape: ', data_test.shape, label_test.shape)


# normalized data
data_train = data_train / 255
data_test = data_test / 255

# init parameters
learning_rate = 0.005
epochs = 1000
losses = torch.zeros(epochs) # tensor to save losses for print


# design model
in_features = data_train.shape[1]
print(in_features)
layer1 = nn.Linear(in_features,20)
layer2 = nn.Linear(20,5)
layer3 = nn.Linear(5,1)
# Xavier (Glorot) initialization
# init.xavier_uniform_(layer1.weight)
# init.xavier_uniform_(layer2.weight)
# init.xavier_uniform_(layer3.weight)
# Kaiming (He) initialization
# init.kaiming_normal_(layer1.weight, nonlinearity='relu')
# init.kaiming_normal_(layer2.weight, nonlinearity='relu')
# init.kaiming_normal_(layer3.weight, nonlinearity='sigmoid')
model = nn.Sequential(
    layer1,
    nn.ReLU(),
    layer2,
    nn.ReLU(),
    layer3,
    nn.Sigmoid()
)

#construct loss and optimizer
Loss = nn.BCELoss()

# init optimizer
optim = torch.optim.Adam(model.parameters(), lr=learning_rate)

# training loop

for epoch in range(epochs):
    # forward
    y_predict = model(data_train)

    # backward
    optim.zero_grad()
    loss = Loss(y_predict, label_train)
    loss.backward()

    if epoch % 10 == 0:
        print(f"epoch= {epoch} loss={loss.item():.4f} ")
    
    # update wights
    optim.step()  
    losses[epoch]=loss
    

# print results
plt.plot(losses.detach(), 'o')
plt.show()

# Test
threshold = 0.7
with torch.no_grad():
    y_predicted = model(data_test)
    y_predicted_bin = y_predicted > threshold
    accuracy = y_predicted_bin.eq(label_test).sum() / float(label_test.shape[0])
    print (f'accuracy = {accuracy:.4f}')

with torch.no_grad():
    y_predicted = model(data_train)
    y_predicted_bin = y_predicted > threshold
    accuracy = y_predicted_bin.eq(label_train).sum() / float(label_train.shape[0])
    print (f'accuracy = {accuracy:.4f}')

