import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import torch
import os
from train import load_trainset, save_fvs, run
from fvs_query import load_testset, save_fvsquery
from embedding import tsne, tsne_centroid
from similarity import cal_score, cal_score_tsne
import argparse


def load_fvs(version, fvs_path, arg_data):
    # normal
    fvs = np.load(fvs_path + 'fvs_{}_ver{}.npy'.format(arg_data, version))
    label = np.load(fvs_path + 'label_{}_ver{}.npy'.format(arg_data, version))
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


def score_of_query(fvs, labels, fvs_query, label_query, arg_dim):
    scores_all = []
    df_labels = pd.DataFrame(labels, columns=['label'])
    for fq, lq in zip(fvs_query, label_query):
        fvs_yhat = fvs[df_labels['label'] == int(lq), :]
        label_yhat = labels[df_labels['label'] == int(lq), :]
        print('pred : {}'.format(int(lq)))

        fvs_concat = np.vstack([fvs_yhat, fq])
        label_concat = np.vstack([label_yhat, lq])

        if arg_dim == 'nd':
            centroid = np.mean(fvs, axis=0)
            similarity_scores = cal_score(centroid, fvs_concat)
            threshold = np.min(similarity_scores[:-1])
            score_query = similarity_scores[-1]
            if score_query >= threshold:
                print('score of query : {}, threshold : {}'.format(np.around(score_query, 4), np.around(threshold, 4)))
                print('similarity between query_img and normal_img of class{} : similar'.format(lq))
            else:
                print('score of query : {}, threshold : {}'.format(np.around(score_query, 4), np.around(threshold, 4)))
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
    return scores_all, score_query


if __name__ == "__main__":
    version = 0  # 조정 부분
    run()
    num_epochs = 5  # 조정 부분

    # parser 생성
    parser = argparse.ArgumentParser()

    # 인자 조건 설정
    parser.add_argument('--data', type=str, default='both',
                        choices=['both', 'normal'],
                        help='what is the data needed in your task?')
    parser.add_argument('--dim', type=str, default='nd',
                        choices=['nd', '2d'],
                        help='what is the data needed in your task?')

    # parsing 후 저장
    args = parser.parse_args()
    arg_data = args.data
    arg_dim = args.dim

    # n_classes 할당
    if arg_data == 'both':
        n_classes = 4   # 조정 부분
    else:
        n_classes = 2   # 조정 부분

    # train 후 fvs 저장
    fvs_path = './fvs/fvs_{}_ver{}/'.format(arg_data, version)
    weight_path = './weights/weights_{}_ver{} (epochs={}).pth'.format(arg_data, version, num_epochs)
    if not (os.path.isdir(fvs_path) and os.path.isfile(weight_path)):
        trainset, train_loader = load_trainset(arg_data)

        trainset_sizes = {x: len(trainset[x]) for x in ['train', 'val']}
        class_names = trainset['train'].classes
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        save_fvs(train_loader, trainset_sizes, class_names, n_classes, device, arg_data, version, num_epochs)

    # fvs_query 저장
    test_dir = './data/query_img/'
    testset, test_loader = load_testset(test_dir)

    save_fvsquery(test_loader, n_classes, arg_data, version, num_epochs)

    # score 계산
    fvs, labels, fvs_query, label_query = load_fvs(version, fvs_path, arg_data)
    scores_all, score_query = score_of_query(fvs, labels, fvs_query, label_query, arg_dim)

    # box-plot
    fig, axes = plt.subplots(1, 4)
    for i, scores in enumerate(scores_all):
        axes[i].boxplot(scores)
        axes[i].axhline(scores[-1])
        axes[i].set_title('Similarity_scores : {}'.format(round(scores[-1], 4)))
        axes[i].set_xlabel('class')
        if i == 0:
            axes[i].set_ylabel('Similarity_scores')

    plt.show()







