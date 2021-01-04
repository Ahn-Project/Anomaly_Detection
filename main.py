import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
from embedding import tsne, tsne_centroid
from similarity import cal_score, box_plot


def load_normal(version, fvs_path):
    # normal
    fvs = np.load(fvs_path + 'fvs_DAGM2007_ver{}.npy'.format(version))
    label = np.load(fvs_path + 'label_DAGM2007_ver{}.npy'.format(version))
    label = label.reshape((-1, 1))
    print(fvs.shape)
    print(label.shape)

    # # abnormal
    # fvs_abnormal = np.load(fvs_path + 'fvs_DAGM2007_abnormal_ver{}.npy'.format(version))
    # label_abnormal = np.load(fvs_path + 'label_DAGM2007_abnormal_ver{}.npy'.format(version))
    # label_abnormal_trfm = list(map(lambda x: int(label_abnormal.tolist()[x]) + 2, range(len(label_abnormal))))
    # label_abnormal_trfm = np.array(label_abnormal_trfm).reshape((-1, 1))
    # print(fvs_abnormal.shape)
    # print(label_abnormal.shape)

    #query_img
    # fvs_query_path = './fvs/fvs_query_ver{}/'.format(version)
    fvs_query = np.load(fvs_path + 'fvs_query_ver{}.npy'.format(version))
    label_query = np.load(fvs_path + 'label_query_ver{}.npy'.format(version))
    label_query = label_query.reshape((-1, 1))
    print(fvs_query.shape)
    print(label_query.shape)

    #
    fvs_concat = np.vstack([fvs, fvs_query])
    label_concat = np.vstack([label, label_query])
    # fvs_concat = np.vstack([fvs, fvs_abnormal, fvs_query])
    # label_concat = np.vstack([label, label_abnormal_trfm, label_query])
    print(fvs_concat.shape)
    print(label_concat.shape)
    return fvs_concat, label_concat, fvs_query, label_query


if __name__ == "__main__":
    ### Load
    version = 0
    fvs_path = './fvs/fvs_normal_ver{}/'.format(version)

    fvs, labels, fvs_query, label_query = load_normal(version, fvs_path)
    df_labels = pd.DataFrame(labels, columns=['label'])

    for pred in label_query:
        fvs_yhat = fvs[df_labels['label'] == int(pred), :]
        label_yhat = labels[df_labels['label'] == int(pred), :]
        print('pred : {}'.format(pred))
        print(fvs_yhat.shape)
        print(label_yhat.shape)

    #
    # # embedding using t-SNE
    # tsne_fit = tsne(fvs_yhat)
    # # print(tsne_fit)
    #
    # centroid = tsne_centroid(tsne_fit)
    # similarity_scores = cal_score(centroid, tsne_fit)
    # boxplot = box_plot(similarity_scores)





