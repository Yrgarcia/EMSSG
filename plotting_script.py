import matplotlib.pyplot as plt
import sys


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


# plot_spearmans(sys.argv[1])
plot_spearmans("LOG_not_enr_senses")
plot_spearmans("LOG_not_enr_context_embs")

# plot_spearmans("LOG_ENR")
# plot_spearmans("LOG_SG")
