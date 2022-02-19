import torch.utils.data as data
from torchvision import datasets, models, transforms
import torch


class experimental_dataset(data.Dataset):

    def __init__(self, data, transform):
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data.shape[0])

    def __getitem__(self, idx):
        item = self.data[idx]
        item = self.transform(item)
        return item


transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor()
])

x = torch.rand(4, 1, 2, 2)
print(x)
epoch = 20

dataset = experimental_dataset(x, transform)
for i in range(epoch):
    for item in dataset:
        print("--------------------")
        print(item)
