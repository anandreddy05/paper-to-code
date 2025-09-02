import torch
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
from torch.utils.data import Subset
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_transform = transforms.Compose([
    transforms.Resize(224),
    transforms.RandomHorizontalFlip(p=0.3),
    transforms.ToTensor()
])
test_tranform = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor()
])


train_data = CIFAR10(root="data",
                     train=True,
                     transform=train_transform,
                     download=True)
test_data = CIFAR10(root="data",
                    train=False,
                    transform=test_tranform,
                    download=True)


# Keep only first 2000 training samples
small_train = Subset(train_data, range(2000))

# Keep only first 300 test samples
small_test = Subset(test_data, range(300))

# Using only a small part of data
BATCH_SIZE = 32
train_dataloader = DataLoader(dataset=small_train,
                              batch_size=BATCH_SIZE,
                              shuffle=True,
                              num_workers=os.cpu_count() or 1)
test_dataloader = DataLoader(dataset=small_test,
                             batch_size=BATCH_SIZE,
                             num_workers=os.cpu_count() or 1,
                             )

class_names = train_data.classes



# BATCH_SIZE = 32
# train_dataloader = DataLoader(dataset=train_data,
#                               batch_size=BATCH_SIZE,
#                               shuffle=True,
#                               num_workers=os.cpu_count() or 1)
# test_dataloader = DataLoader(dataset=test_data,
#                              batch_size=BATCH_SIZE,
#                              num_workers=os.cpu_count() or 1,
#                              )