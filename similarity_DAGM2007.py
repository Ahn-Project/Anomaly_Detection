# -*- coding: utf-8 -*-
"""feature_vector_clustering (with hymenoptera).ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1C0vc9aZUObC8_2B3JOg4_l6UjUVAvBis
"""


###########################################
# TSNE (for Dimensionality Reduction) & grouping with colors
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import os


def euclidean_similarity(v1, v2):
  from scipy.spatial import distance

  dst = distance.euclidean(v1, v2)
  similarity_score = 1/(1+dst)
  return similarity_score


def tsne(fvs, label):
  tsne = TSNE()
  tsne_fit = tsne.fit_transform(fvs[:len(fvs)])

  tsne_x = pd.DataFrame(tsne_fit[:, 0], columns=['tsne_x'])
  tsne_y = pd.DataFrame(tsne_fit[:, 1], columns=['tsne_y'])
  df_tsne = pd.concat([tsne_x, tsne_y], axis=1)
  df_label = pd.DataFrame(label, columns=['label'])
  data = pd.concat([df_label, df_tsne],axis=1)

  # plt.scatter(tsne_fit[:, 0], tsne_fit[:, 1])

  colors = 'b', 'r', 'y', 'black'
  plt_labels = 'Normal_class1', 'Normal_class2', 'Abnormal_class1', 'Abnormal_class2'
  for i, c, label in zip(range(len(plt_labels)), colors, plt_labels):
        plt.scatter(tsne_fit[data['label']==i, 0], tsne_fit[data['label']==i, 1], c=c, label=label, alpha=0.5)
  plt.title('TSNE of FVs from 2classes_classifier_{} (seperated by colors)'.format(version))
  plt.legend()
  plt.show()

  fig = plt.gcf()
  # fig.savefig(os.path.join(fvs_path, plt_title+'.png'), dpi=fig.dpi)
  return tsne_fit, data, plt_labels


# 유사도 계산
def cal_score(tsne_fit, data, pair_labels_order):
  similarity_scores = []
  for i, order in enumerate(pair_labels_order):
      tempx = tsne_fit[data['label'] == order, 0]
      tempy = tsne_fit[data['label'] == order, 1]
      # plt.scatter(tempx, tempy, label=str(i), alpha=0.5)

      if i%2 == 0:
          mean_x = np.mean(tempx)
          mean_y = np.mean(tempy)
          mean_xy = np.array([mean_x, mean_y])  # 유사도 기준 벡터 정의
          # plt.scatter(mean_x, mean_y, c='r', s=50)

      scores = []
      tsne_xy = np.stack((tempx, tempy), axis=1)  # 유사도 대상 벡터 정의

      for xy in tsne_xy:
          score = euclidean_similarity(mean_xy, xy)  # 모든 대상 벡터에 대한 유사도 계산
          scores.append(score)

      similarity_scores.append(scores)  # 클래스별 유사도 append

  return similarity_scores



if __name__ == "__main__":
  ### Load
  version = 'ver4'
  fvs_path = './fvs/2classes_classifier_(using_Normal)_{}/'.format(version)
  pair_labels_order = [0, 2, 1, 3]

  # normal
  fvs = np.load(fvs_path + 'fvs_DAGM2007_{}.npy'.format(version))
  label = np.load(fvs_path + 'label_DAGM2007_{}.npy'.format(version))
  print(fvs.shape)
  print(label.shape)

  # abnormal
  fvs_abnormal = np.load(fvs_path + 'fvs_DAGM2007_abnormal_{}.npy'.format(version))
  label_abnormal = np.load(fvs_path + 'label_DAGM2007_abnormal_{}.npy'.format(version))
  label_abnormal_trfm = list(map(lambda x : int(label_abnormal.tolist()[x]) + 2, range(len(label_abnormal))))
  label_abnormal_trfm = np.array(label_abnormal_trfm)
  print(fvs_abnormal.shape)
  print(label_abnormal.shape)

  # concat
  fvs_concat = np.vstack([fvs,fvs_abnormal])
  label_concat = np.vstack([label.reshape((-1, 1)),label_abnormal_trfm.reshape((-1, 1))])
  print(fvs_concat.shape)
  print(label_concat.shape)

  # 순서 바꾸기
  df_label_concat = pd.DataFrame(label_concat, columns=['label'])

  # from collections import Counter
  # len_label = len(Counter(label_concat.reshape((len(label_concat)))))

  for i in pair_labels_order:
    if i == 0:
      fvs_pair = fvs_concat[df_label_concat['label'] == i]
      label_pair = label_concat[df_label_concat['label'] == i]
    else:
      fvs_pair = np.vstack((fvs_pair, fvs_concat[df_label_concat['label'] == i]))
      label_pair = np.vstack((label_pair, label_concat[df_label_concat['label'] == i]))

  ####################################
  tsne_fit, data, plt_labels = tsne(fvs_pair, label_pair)
  similarity_scores = cal_score(tsne_fit, data, pair_labels_order)

  from collections import Counter
  for i, label in enumerate(pair_labels_order):
    max_score = np.max(similarity_scores[i])
    min_score = np.min(similarity_scores[i])
    print('highest similarity score of {} : {}'.format(plt_labels[label], max_score))
    print('lowest similarity score of {} : {}'.format(plt_labels[label], min_score))

    if i % 2 == 0:
      min_score_normal = min_score
    elif i % 2 == 1:
      num_misclassified = Counter(min_score_normal < np.array(similarity_scores[i]))
      keys = num_misclassified.keys()
      if True in keys:
        print('misclassification rate of {} : {} / {}'.format(label, dict(num_misclassified)[True],
                                                              len(similarity_scores[i])))
        print('=' * 20)
      else:
        print('misclassification rate of {} : {} / {}'.format(label, 0, len(similarity_scores[i])))
        print('=' * 20)

