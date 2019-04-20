# -*- coding: utf-8 -*-
"""
@author: Yoalli García
"""

import matplotlib.pyplot as plt
import numpy as np


def plot_spearmans(file):
    # plot spearman correlations from file containing one score per line
    with open(file) as myfile:
        allData = myfile.readlines()
    ii = []
    all_data = []
    for i, data in enumerate(allData):
        # plt.scatter(i, float(data))
        ii.append(i)
        all_data.append(float(data))
    plt.plot(ii, all_data, '-o')
    plt.title(file)
    plt.show()


def create_sense_dict(sense_files):
    # load all embeddings from sense_files
    sense_dict = {}
    for i in range(len(sense_files)):
        with open(sense_files[i]) as sf:
            lines = sf.readlines()
        for line in lines:
            temp_dict = {}
            line_list = line.split(" ")
            if i == 0:
                temp_dict[i] = np.array([float(x) for x in line_list[1:]])
                sense_dict[line_list[0]] = temp_dict
            else:
                sense_dict[line_list[0]][i] = np.array([float(x) for x in line_list[1:]])
    return sense_dict


def plot_senses(sense_dict, title, limit=100):
    # plot 2D embeddings to compare senses for testing purposes
    V = []
    plot_dict = {}
    cmap = plt.get_cmap('jet')
    # colors = cmap(np.linspace(0, 1.0, len(sense_dict.keys())))
    colors = cmap(np.linspace(0, 1.0, limit))
    markers = [".", "o", "v", "^", "<", ">", "8", "s", "p", "P", "*", "h", "H", "+", "x", "X", "D", "d"]
    i = 0
    while markers.__len__() != len(sense_dict.keys()):
        if i > len(markers):
            i = 0
        markers.append(markers[i])
        i += 1
    #print(markers)
    i = 0
    for word in sense_dict:
        if i < limit:
            plot_dict[word] = {}
            y = []
            x = []
            for sense in sense_dict[word]:
                t = sense_dict[word][sense]
                V.append(t)
                x.append(t[0])
                y.append(t[1])
            plot_dict[word]["x"] = x
            plot_dict[word]["y"] = y
            i += 1
    for i, color, marker in zip(plot_dict, colors, markers):
        plt.scatter(plot_dict[i]["x"], plot_dict[i]["y"], label=i, c = color, marker = marker)
    # plt.legend(ncol=2, loc="best")
    plt.title(title)
    plt.legend(ncol=2, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.autoscale()
    plt.tight_layout()
    axes = plt.gca()
    axes.set_xlim([-3, 3])
    axes.set_ylim([-4, 4])
    axes.axhline(y=0, color='k')
    axes.axvline(x=0, color='k')
    plt.show()


def plot_embeddings(emb_matrix, words, name):
    # function for visualizing sense vectors and their nearest global or sense vectors
    U, s, Vh = np.linalg.svd(emb_matrix, full_matrices=False)
    print(U[0, 0], U[0, 1])
    print(U[1, 0], U[1, 1])
    for i in range(len(words)):
        fig = plt.gcf()
        fig.set_size_inches(18.5, 10.5)
        plt.text(U[i, 0], U[i, 1], words[i])
        plt.xlim((-1.0, 1.0))
        plt.ylim((-1.0, 1.0))
    plt.title(name)
    plt.show()


if __name__ == '__main__':
    pass
