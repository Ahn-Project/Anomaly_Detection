import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
from embedding import tsne, tsne_centroid
from similarity import cal_score


def load_normal(version, fvs_path):
    # normal
    fvs = np.load(fvs_path + 'fvs_both_ver{}.npy'.format(version))
    label = np.load(fvs_path + 'label_both_ver{}.npy'.format(version))
    label = label.reshape((-1, 1))
    # print(fvs.shape)
    # print(label.shape)

    #query_img
    fvs_query = np.load(fvs_path + 'fvs_query_ver{}.npy'.format(version))
    label_query = np.load(fvs_path + 'label_query_ver{}.npy'.format(version))
    label_query = label_query.reshape((-1, 1))
    # print(fvs_query.shape)
    # print(label_query.shape)

    return fvs, label, fvs_query, label_query


if __name__ == "__main__":
    ### Load
    version = 1
    fvs_path = './fvs/fvs_both_ver{}/'.format(version)

    fvs, labels, fvs_query, label_query = load_normal(version, fvs_path)
    df_labels = pd.DataFrame(labels, columns=['label'])

    scores_all = []
    for fq, lq in zip(fvs_query, label_query):
        fvs_yhat = fvs[df_labels['label'] == int(lq), :]
        label_yhat = labels[df_labels['label'] == int(lq), :]
        print('pred : {}'.format(int(lq)))
        # print(fvs_yhat.shape)
        # print(label_yhat.shape)

        fvs_concat = np.vstack([fvs_yhat, fq])
        label_concat = np.vstack([label_yhat, lq])
        # print(fvs_concat.shape)
        # print(label_concat.shape)

        # embedding using t-SNE
        tsne_fit = tsne(fvs_concat)

        centroid = tsne_centroid(tsne_fit[:-1])
        similarity_scores = cal_score(centroid, tsne_fit)
        threshold = np.min(similarity_scores[:-1])
        score_of_query = similarity_scores[-1]
        if score_of_query >= threshold:
            print('score of query : {}, threshold : {}'.format(np.around(score_of_query, 3), np.around(threshold, 3)))
            print('similarity between query_img and normal_img of class{} : similar'.format(lq))
        else:
            print('score of query : {}, threshold : {}'.format(np.around(score_of_query, 3), np.around(threshold, 3)))
            print('similarity between query_img and normal_img of class{} : different'.format(lq))

        scores_all.append(similarity_scores)

    # box-plot
    fig, axes = plt.subplots(1, 4)
    for i, scores in enumerate(scores_all):
        axes[i].boxplot(scores)
        axes[i].axhline(scores[-1])
        axes[i].set_title('Similarity_scores : {}'.format(round(scores[-1], 3)))
        axes[i].set_xlabel('class')
        if i == 0:
            axes[i].set_ylabel('Similarity_scores')

    plt.show()
        # boxplot = box_plot(similarity_scores, ax)






