import numpy as np
import torch

from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder

class MsiMssCrc(Dataset):
  Train = 0
  Valid = 1

  def __init__(self, path, mode=0, transform=None):
    self.data = ImageFolder(path)
    self.kinds = self.data.classes
    self.transform = transform or (lambda x: x)
    self.mode = mode
    self.size = len(self.data) // 2
    self.train_end = int(self.size * 0.8)
    self.offset = 0 if self.mode == MsiMssCrc.Train else self.train_end
    self.mode_size = self.train_end if self.offset == 0 else self.size - self.offset

  def __getitem__(self, index):
    target_value = index // self.mode_size
    if target_value:
      index = index - self.mode_size + self.size + self.offset
    else:
      index = index + self.offset

    data, target = self.data[index]
    data = np.array(data).transpose(2, 0, 1)
    data = torch.tensor(data, dtype=torch.float)
    return self.transform(data), target

  def __len__(self):
    return 2 * self.mode_size

class GANData(MsiMssCrc):
  def __getitem__(self, index):
    data, label = super().__getitem__(index)
    onehot = torch.zeros(2)
    onehot[label] = 1
    return data, [onehot]
