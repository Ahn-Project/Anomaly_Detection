import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
from embedding import tsne, tsne_centroid
from similarity import cal_score, cal_score_tsne


def load_fvs(version, fvs_path):
    # normal
    fvs = np.load(fvs_path + 'fvs_both_ver{}.npy'.format(version))
    label = np.load(fvs_path + 'label_both_ver{}.npy'.format(version))
    label = label.reshape((-1, 1))
    # print(fvs.shape)
    # print(label.shape)

    # query_img
    fvs_query = np.load(fvs_path + 'fvs_query_ver{}.npy'.format(version))
    label_query = np.load(fvs_path + 'label_query_ver{}.npy'.format(version))
    label_query = label_query.reshape((-1, 1))
    # print(fvs_query.shape)
    # print(label_query.shape)
    return fvs, label, fvs_query, label_query


def score_of_query(fvs, labels, fvs_query, label_query, arg=0):
    scores_all = []
    df_labels = pd.DataFrame(labels, columns=['label'])
    for fq, lq in zip(fvs_query, label_query):
        fvs_yhat = fvs[df_labels['label'] == int(lq), :]
        label_yhat = labels[df_labels['label'] == int(lq), :]
        print('pred : {}'.format(int(lq)))

        fvs_concat = np.vstack([fvs_yhat, fq])
        label_concat = np.vstack([label_yhat, lq])

        if arg == 0:
            centroid = np.mean(fvs, axis=0)
            similarity_scores = cal_score(centroid, fvs_concat)
            threshold = np.min(similarity_scores[:-1])
            score_query = similarity_scores[-1]
            if score_query >= threshold:
                print('score of query : {}, threshold : {}'.format(np.around(score_query, 3), np.around(threshold, 3)))
                print('similarity between query_img and normal_img of class{} : similar'.format(lq))
            else:
                print('score of query : {}, threshold : {}'.format(np.around(score_query, 3), np.around(threshold, 3)))
                print('similarity between query_img and normal_img of class{} : different'.format(lq))

        else:
            # embedding using t-SNE
            tsne_fit = tsne(fvs_concat)

            centroid = tsne_centroid(tsne_fit[:-1])
            similarity_scores = cal_score_tsne(centroid, tsne_fit)
            threshold = np.min(similarity_scores[:-1])
            score_query = similarity_scores[-1]
            if score_query >= threshold:
                print('score of query : {}, threshold : {}'.format(np.around(score_query, 3), np.around(threshold, 3)))
                print('similarity between query_img and normal_img of class{} : similar'.format(lq))
            else:
                print('score of query : {}, threshold : {}'.format(np.around(score_query, 3), np.around(threshold, 3)))
                print('similarity between query_img and normal_img of class{} : different'.format(lq))

        scores_all.append(similarity_scores)
    return scores_all


if __name__ == "__main__":
    ### Load
    version = 1
    fvs_path = './fvs/fvs_both_ver{}/'.format(version)
    arg = 0

    fvs, labels, fvs_query, label_query = load_fvs(version, fvs_path)
    if arg == 0:
        scores_all = score_of_query(fvs, labels, fvs_query, label_query, arg=arg)
    else:
        scores_all = score_of_query(fvs, labels, fvs_query, label_query, arg=arg)

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







