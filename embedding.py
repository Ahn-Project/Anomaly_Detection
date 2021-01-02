import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import pandas as pd
import numpy as np
import os


def tsne(fvs):
    tsne = TSNE()
    tsne_fit = tsne.fit_transform(fvs)
    return tsne_fit


def tsne_visualization(tsne_fit, data, class_names, version):
    # plt.scatter(tsne_fit[:, 0], tsne_fit[:, 1])

    # colors = 'b', 'r', 'y', 'black'
    # plt_labels = 'Normal_class1', 'Normal_class2', 'Abnormal_class1', 'Abnormal_class2'
    plt_labels = class_names
    for i, label in zip(range(len(plt_labels)), plt_labels):
          plt.scatter(tsne_fit[data['label']==i, 0], tsne_fit[data['label']==i, 1], label=label, alpha=0.5)
    plt.title('TSNE of FVs from 2classes_classifier_{} (seperated by colors)'.format(version))
    plt.legend()
    plt.show()

    # fig = plt.gcf()
    # fig.savefig(os.path.join(fvs_path, plt_title+'.png'), dpi=fig.dpi)


def euclidean_similarity(v1, v2):
  from scipy.spatial import distance

  dst = distance.euclidean(v1, v2)
  similarity_score = 1/(1+dst)
  return similarity_score


def tsne_centroid(tsne_fit):
    tsne_x = tsne_fit[:, 0]
    tsne_y = tsne_fit[:, 1]

    mean_x = np.mean(tsne_x)
    mean_y = np.mean(tsne_y)
    centroid = np.array([mean_x, mean_y])
    return centroid




