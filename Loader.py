import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch



file = 'Data/cats.pth'
data, labels = torch.load(file)

print(data.shape, labels.shape)
print(data[25])
# Example of a picture
index1 = 25 # change index to get a different picture
index2 = 26 # change index to get a different picture
plt.subplot(1,2,1)
plt.imshow(data[index1])
plt.title(labels[index1])
plt.subplot(1,2,2)
plt.imshow(data[index2])
plt.title(labels[index2])
plt.show()
