import matplotlib.pyplot as plt
import matplotlib
from matplotlib.font_manager import FontProperties
import numpy as np


def plot_spearmans(file):
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
    # load all embedding files embeddings (sense_files)
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


def plot_senses(sense_dict, title):
    # plot 2D embeddings to compare senses
    V = []
    plot_dict = {}
    cmap = plt.get_cmap('jet')
    colors = cmap(np.linspace(0, 1.0, len(sense_dict.keys())))
    markers = [".", "o", "v", "^", "<", ">", "8", "s", "p", "P", "*", "h", "H", "+", "x", "X", "D", "d"]
    i = 0
    while markers.__len__() != len(sense_dict.keys()):
        if i > len(markers):
            i = 0
        markers.append(markers[i])
        i += 1
    #print(markers)
    for word in sense_dict:
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
    for i, color, marker in zip(plot_dict, colors, markers):
        plt.scatter(plot_dict[i]["x"], plot_dict[i]["y"], label=i, c = color, marker = marker)
    # plt.legend(ncol=2, loc="best")
    plt.title(title)
    plt.legend(ncol=2, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.autoscale()
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    # plot_spearmans(sys.argv[1])
    # plot_spearmans("LOG_not_enr_senses")
    # plot_spearmans("LOG_enr_senses")
    # plot_spearmans("LOG_not_enr_context_embs")
    # plot_spearmans("LOG_enr_context_embs")
    # sense_dict = create_sense_dict(["not_enr_SENSES_0", "not_enr_SENSES_1"])
    # my_dict = create_sense_dict(["not_enr_MYS_0", "not_enr_MYS_1"])
    # ctxt_dict = create_sense_dict(["MSSG-TEST_en-5-2-2"])
    # plot_senses(sense_dict, "Sense Vectors")
    # plot_senses(ctxt_dict, "Context Vectors")
    # plot_senses(my_dict, "My Cluster Centers")
    # plot_spearmans("LOG_ENR")
    # plot_spearmans("LOG_SG")
    pass
