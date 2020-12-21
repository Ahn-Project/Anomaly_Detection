# -*- coding: utf-8 -*-
"""train (similarity_based)_with real.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1La4YIl7xThrTnKYfiStsXkiDYJ5_fdFp
"""


# from __future__ import print_function, division
from model import resnet
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import pandas as pd
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy



# real image 데이터 로드
data_transforms = {
    'train' : transforms.Compose([
                    transforms.Resize(224),
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomVerticalFlip(),
                    # transforms.RandomRotation([45,90,180,270]),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456,0.406],[0.229,0.224,0.225])              
    ]),
    'val' : transforms.Compose([
                    transforms.Resize(224),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])
}

batch_size = 4  # 조정 부분
root_dir = './data/subset2/'    # 조정 부분
dataset = {x : datasets.ImageFolder(os.path.join(root_dir, x),
                                    transform = data_transforms[x]) for x in ['train','val']}

data_loader = {x : torch.utils.data.DataLoader(dataset[x],
                                               batch_size = batch_size,
                                               shuffle = True) for x in ['train','val']}


dataset_sizes = {x : len(dataset[x]) for x in ['train', 'val']}
class_names = dataset['train'].classes
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


##########################################################
def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
  since = time.time()

  best_model_wts = copy.deepcopy(model.state_dict())
  best_acc = 0.0

  fvs,label_epoch = [],[]  # 추가 부분
  for epoch in range(num_epochs):
    for phase in ['train', 'val']:
      if phase == 'train':
        model.train()
      else:
        model.eval()

      running_loss = 0.0
      running_correct = 0
      for inputs, labels in data_loader[phase]:
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        with torch.set_grad_enabled(phase=='train'):
          outputs,fv = model(inputs)  # 수정 부분
          _, preds = torch.max(outputs, 1)
          loss = criterion(outputs, labels)
          if phase == 'val' and 1<=epoch<3:
            fvs.extend(fv) # 추가 부분
            label_epoch.extend(labels) # 추가 부분

          if phase == 'train':
            loss.backward()
            optimizer.step()

        running_loss += loss.item()*inputs.size(0)
        running_correct += torch.sum(preds==labels.data)

      if phase == 'train':
        scheduler.step()

      epoch_loss = running_loss / dataset_sizes[phase]
      epoch_acc = running_correct / dataset_sizes[phase]

      print('{} Loss: {:.4f} Acc: {:.4f}'.format(
              phase, epoch_loss, epoch_acc))

      if phase == 'val' and epoch_acc > best_acc:
        best_acc = epoch_acc
        best_model_wts = copy.deepcopy(model.state_dict())
    print()

  time_elapsed = time.time() - since
  print('Training complete in {:.0f}m {:.0f}s'.format(
      time_elapsed // 60, time_elapsed % 60))
  print('Best val Acc: {:4f}'.format(best_acc))

  # 가장 나은 모델 가중치를 불러옴
  model.load_state_dict(best_model_wts)
  return model, best_acc, fvs, label_epoch

def run():
      torch.multiprocessing.freeze_support()
      print('loop')


##########################################################
### weight Downloading: "https://download.pytorch.org/models/resnet18-5c106cde.pth" to C:\Users\neouly08/.cache\torch\hub\checkpoints\resnet18-5c106cde.pth

model_ft = resnet.resnet18(pretrained = True)
# model_ft = models.resnet18(pretrained = True)
num_ftrs = model_ft.fc.in_features
model_ft.fc = nn.Linear(num_ftrs, 24)   # 조정 부분

model_ft = model_ft.to(device)

criterion = nn.CrossEntropyLoss()

optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

if __name__ == '__main__':
      version = 'ver2'  # 조정 부분

      run()
      num_epochs = 5
      model_ft, best_acc, fvs, label_epoch = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler, num_epochs=num_epochs)

      wts_save_path = './weights/'
      torch.save(model_ft.state_dict(), wts_save_path+'weights_counterfeit_{} (epochs={}).pth'.format(version, num_epochs))

      ##########################################################
      ### 추가 부분 (for clustering)
      fvs_list = list(map(lambda x: fvs[x].tolist(), range(len(fvs))))  # 추가 부분
      label_item = list(map(lambda x: label_epoch[x].item(), range(len(label_epoch))))  # 추가 부분

      # 추가 부분
      fvs_array = np.array(fvs_list[:])
      label_array = np.array(label_item[:])
      # print(fvs_60000.shape)
      # print(label_60000.shape)

      fvs_save_path = './fvs_clustering/'
      dir_path = fvs_save_path + '{}'.format(version)
      if not os.path.isdir(dir_path):
          os.mkdir(dir_path)

      np.save(os.path.join(dir_path, 'fvs_counterfeit_{}'.format(version)), fvs_array)
      np.save(os.path.join(dir_path, 'label_counterfeit_{}'.format(version)), label_array)

      # np.save(fvs_save_path + 'fvs_DAGM2007_normal_and_abnormal_{}'.format(version), fvs_array)
      # np.save(fvs_save_path + 'label_DAGM2007_normal_and_abnormal_{}'.format(version), label_array)

