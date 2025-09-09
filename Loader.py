import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch



file = 'Data/cats.pth'
data, labels = torch.load(file)

print(data.shape, labels.shape)
print(labels)
print(data[25])
# Example of a picture

index1 = 25 # change index to get a different picture
index2 = 20 # change index to get a different picture
img1 = data[index1].astype('uint8')
img2 = data[index2].astype('uint8')
plt.subplot(1,2,1)
plt.imshow(img1)
plt.title(labels[index1])
plt.subplot(1,2,2)
plt.imshow(img2)
plt.title(labels[index2])
plt.show()
