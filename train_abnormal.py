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
import os
import copy

# abnormal image 데이터 로드
data_transforms = transforms.Compose([
                    transforms.Resize(224),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
                    ])


root_dir = './data/DAGM 2007 Dataset/abnormal/'
dataset = datasets.ImageFolder(os.path.join(root_dir, 'val'),
                                transform = data_transforms)

testloader = torch.utils.data.DataLoader(dataset,
                                          batch_size = 4,
                                          shuffle = False) 


dataset_sizes = len(dataset)
class_names = dataset.classes
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# print(dataset_sizes['train'])


##########################################################
def load_model(wts_name):
      weights_path = './weights/' + wts_name 
      model = resnet.resnet18(pretrained = False)
      num_ftrs = model.fc.in_features
      model.fc = nn.Linear(num_ftrs, 2)
      model.load_state_dict(torch.load(weights_path))
      return model

def test(model, testLoader):
    correct = 0
    total = 0
    # tp, fp, tn, fn = 0,0,0,0
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    fvs,label_epoch = [],[]  # 추가 부분
    with torch.no_grad():
        for i, data in enumerate(testLoader):
            inputs, labels = data[0].to(device), data[1].to(device)
            outputs,fv = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            fvs.extend(fv) # 추가 부분
            label_epoch.extend(labels) # 추가 부분
            # pred_argmax = torch.argmax(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            print('iter:{}  accuarcy:{}'.format(i,(correct / total)))

    #         tp_i, fp_i, tn_i, fn_i = confusion_matrix(pred_argmax, labels)
    #         tp += tp_i
    #         fp += fp_i
    #         tn += tn_i
    #         fn += fn_i
                        
    accuracy = (correct / total)
    # precision = np.mean(tp/(tp+fp+1e-12))
    # recall = np.mean(tp/(tp+fn+1e-12))
    # f1_score = tp/(tp+(0.5*(fp+fn)))
    print('Accuracy: {}'.format(accuracy))
    # print('Precision: {}'.format(precision))
    # print('Recall: {}'.format(recall))
    # print('F1_score: {}'.format(f1_score))
    # print()
    return accuracy, fvs, label_epoch




##########################################################
### weight Downloading: "https://download.pytorch.org/models/resnet18-5c106cde.pth" to C:\Users\neouly08/.cache\torch\hub\checkpoints\resnet18-5c106cde.pth



if __name__ == "__main__":
      version = 'ver4'

      # load model
      wts_name = 'weights_2classes_{} (epochs=5).pth'.format(version)
      model = load_model(wts_name)
      model.eval()
      # print(model)

      if torch.cuda.is_available():
          model.cuda()

      # test
      accuracy, fvs, label_epoch = test(model, testloader)


      ### 추가 부분 (for clustering)
      fvs_list = list(map(lambda x: fvs[x].tolist(), range(len(fvs))))  # 추가 부분
      label_item = list(map(lambda x: label_epoch[x].item(), range(len(label_epoch))))  # 추가 부분

      # 추가 부분
      fvs_array = np.array(fvs_list[:])
      label_array = np.array(label_item[:])

      fvs_save_path = './fvs_clustering/'
      dir_path = fvs_save_path + '2classes_classifier_(using_Normal)_{}'.format(version)
      if not os.path.isdir(dir_path):
          os.mkdir(dir_path)

      np.save(os.path.join(dir_path, 'fvs_DAGM2007_abnormal_{}'.format(version)), fvs_array)
      np.save(os.path.join(dir_path, 'label_DAGM2007_abnormal_{}'.format(version)), label_array)

      # np.save(fvs_save_path + 'fvs_DAGM2007_abnormal_{}'.format(version), fvs_array)
      # np.save(fvs_save_path + 'label_DAGM2007_abnormal_{}'.format(version), label_array)

