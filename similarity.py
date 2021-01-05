import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def euclidean_similarity(v1, v2):
  from scipy.spatial import distance

  dst = distance.euclidean(v1, v2)
  similarity_score = 1/(1+dst)
  return similarity_score


# 유사도 계산
def cal_score(centroid, tsne_fit):
    similarity_scores = []
    for xy in tsne_fit:
        score = euclidean_similarity(centroid, xy)  # 모든 대상 벡터에 대한 유사도 계산
        similarity_scores.append(score)
    # centroid_v = tsne_fit[np.argmax(similarity_scores)] # centroid와 가장 가까운 벡터 idx 찾기
    return similarity_scores


def box_plot(scores):
    fig, ax = plt.subplots()
    ax.boxplot(scores)
    plt.axhline(scores[-1])
    plt.title('Similarity_scores : {}'.format(round(scores[-1], 3)))
    ax.set_xlabel('class')
    ax.set_ylabel('Similarity_scores')

    plt.show()

