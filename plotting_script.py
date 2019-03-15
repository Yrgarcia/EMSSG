import matplotlib.pyplot as plt
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


def plot_senses(sense_dict, title, limit=100):
    # plot 2D embeddings to compare senses
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


def plot_contexts_for_word(sense_dict, context_dict, title, s0, s1, target):
    # plot 2D embeddings to view contexts of most frequent context words per sense of word
    ctxt_sense_0 = []
    ctxt_sense_1 = []
    for i in range(len(s0)):
        ctxt_sense_0.append(s0[i][0])
        ctxt_sense_1.append(s1[i][0])
    V = []
    plot_dict0 = {}
    plot_dict1 = {}
    cmap = plt.get_cmap('jet')
    num = len(ctxt_sense_0) + len(ctxt_sense_1) + 2
    # colors = cmap(np.linspace(0, 1.0, len(sense_dict.keys())))
    colors = cmap(np.linspace(0, 1.0, num))
    markers = [".", "o", "v", "^", "<", ">", "8", "s", "p", "P", "*", "h", "H", "+", "x", "X", "D", "d"]
    i = 0
    while markers.__len__() != len(sense_dict.keys()):
        if i > len(markers):
            i = 0
        markers.append(markers[i])
        i += 1
    # print(markers)
    plot_dict0[target+"0"] = {}
    plot_dict1[target+"1"] = {}
    plot_dict0[target+"0"]["x"] = sense_dict[target][0][0]
    plot_dict1[target+"1"]["x"] = sense_dict[target][1][0]
    print(sense_dict[target][0][0], sense_dict[target][1][0])
    plot_dict0[target+"0"]["y"] = sense_dict[target][0][1]
    plot_dict1[target+"1"]["y"] = sense_dict[target][1][1]
    for word in context_dict:
        if word in ctxt_sense_1:
            p_word = word+"1"
            plot_dict1[p_word] = {}
            y = context_dict[word][0][1]
            x = context_dict[word][0][0]
            plot_dict1[p_word]["x"] = x
            plot_dict1[p_word]["y"] = y
        if word in ctxt_sense_0:
            p_word = word+"0"
            plot_dict0[p_word] = {}
            y = context_dict[word][0][1]
            x = context_dict[word][0][0]
            plot_dict0[p_word]["x"] = x
            plot_dict0[p_word]["y"] = y
    for i, marker in zip(plot_dict0, markers):
        plt.scatter(plot_dict0[i]["x"], plot_dict0[i]["y"], label=i, c="b", marker=marker)
    for i, marker in zip(plot_dict1, markers):
        plt.scatter(plot_dict1[i]["x"], plot_dict1[i]["y"], label=i, c="r", marker=marker)
    # plt.legend(ncol=2, loc="best")
    plt.title(title)
    plt.legend(ncol=2, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.autoscale()
    plt.tight_layout()
    axes = plt.gca()
    axes.set_xlim([-4, 4])
    axes.set_ylim([-4, 4])
    axes.axhline(y=0, color='k')
    axes.axvline(x=0, color='k')
    plt.show()


def get_n_most_common_ctxt_words(ctxt_dict, topnum):
    import operator
    sorted_x = sorted(ctxt_dict.items(), key=operator.itemgetter(1))
    sorted_x.reverse()
    print(sorted_x[:topnum])
    return sorted_x[:topnum]


def plot_contexts(target, ctxt_words_count0, ctxt_words_count1, ctxt_embs):
    m_common_0 = get_n_most_common_ctxt_words(ctxt_words_count0, 20)
    m_common_1 = get_n_most_common_ctxt_words(ctxt_words_count1, 20)
    sense_dict = create_sense_dict(["not_enr_SENSES_0", "not_enr_SENSES_1"])
    ctxt_dict = create_sense_dict([ctxt_embs])
    plot_contexts_for_word(sense_dict, ctxt_dict, "ASK with contexts", m_common_0, m_common_1, target)


if __name__ == '__main__':
    # plot_spearmans(sys.argv[1])
    # plot_spearmans("LOG_not_enr_senses")
    # plot_spearmans("LOG_enr_senses")
    # plot_spearmans("LOG_not_enr_context_embs")
    # plot_spearmans("LOG_enr_context_embs")
    # ask_ctxt_words_sense_0 = {'reconsider': 1, 'commissioner': 3, 'abstentions': 1, 'examine': 1, 'vice-president': 1, 'halfway': 1, 'presidency': 1, 'agree': 1, 'members': 1, 'bring': 1, 'responsible': 1, 'two': 1, 'portuguese': 1, 'full': 1, 'particularly': 1, 'president': 2, 'reassure': 1, 'request': 2, 'fellow': 1, 'make': 2, 'use': 1}
    # ask_ctxt_words_sense_1 = {'tobin': 15, 'intention': 11, '23': 7, 'understands': 5, 'thing': 30, 'could': 23, 'next': 15, 'sea': 11, 'agree': 17, 'case': 5, 'contact': 15, 'parliamentary': 15, 'full': 10, 'olaf': 8, 'expressly': 15, 'sessional': 15, 'able': 5, 'purpose': 13, 'free': 15, 'portuguese': 14, 'last': 6, 'confirm': 5, 'citizens': 25, 'halfway': 14, 'whether': 120, 'explained': 10, 'criteria': 7, 'thought': 9, 'habit': 6, 'prodi': 39, 'övp': 15, 'affected': 12, 'said': 15, 'vote': 6, 'likely': 10, 'chamber': 15, 'common': 5, 'standards': 5, 'special': 6, 'stop': 12, 'reassure': 12, 'monitor': 9, 'ireland': 15, 'structure': 15, 'second': 21, 'one': 86, 'council': 89, 'incorporate': 11, 'lot': 15, '``': 15, 'parliament': 50, 'time': 15, '2000': 15, 'responsible': 14, 'minister': 15, 'speeches': 4, 'europeans': 15, 'interinstitutional': 15, 'merely': 15, 'back': 9, 'head': 7, 'repeat': 8, "'what": 9, 'committee': 15, 'brief': 15, 'rating': 15, 'verify': 15, 'effect': 12, 'simply': 45, 'analysis': 3, 'reasons': 11, 'code': 15, 'review': 15, 'stage': 7, 'prevent': 4, 'approve': 27, 'europe': 15, 'really': 10, 'legislative': 15, 'financial': 7, 'opinion': 4, 'others': 15, 'commissioner': 162, 'issue': 15, 'clearly': 15, 'programme': 15, 'quaestors': 11, 'life-time': 15, 'ensure': 37, 'two': 5, 'accordance': 15, 'even': 13, 'example': 8, 'fellow': 14, 'accept': 15, 'perfectly': 15, 'referenda': 8, 'respect': 5, 'steps': 15, 'terribly': 5, 'improperly': 15, 'war': 4, 'study': 32, 'eu': 3, 'granting': 9, 'want': 30, 'british': 6, 'motion': 15, 'power': 15, 'times': 3, 'obligation': 15, 'directive': 9, 'december': 11, 'previous': 8, 'pay': 10, 'scientists': 15, 'happen': 15, 'however': 38, 'street': 15, 'president': 29, 'solidarity': 5, 'give': 26, 'destroyed': 7, '1997': 11, 'particularly': 19, 'committees': 9, 'look': 10, 'anti-racism': 5, 'takes': 15, 'may': 45, 'member': 15, 'treaty': 6, 'presidency': 14, 'seems': 4, 'work': 7, 'never': 15, 'grasp': 12, 'accuracy': 7, 'vitally': 5, 'timely': 15, 'money': 27, 'board': 3, 'adopt': 9, 'swiftly': 9, 'regions': 19, 'areas': 15, 'question': 78, 'mr': 95, 'union': 30, 'please': 8, 'enough': 15, 'right': 19, 'years': 7, 'floor': 11, 'commission': 298, 'base': 4, 'tomorrow': 15, 'safety': 6, 'abstentions': 10, 'report': 36, 'services': 13, 'resources': 12, 'clarification': 15, 'take': 60, 'furthermore': 8, 'support': 65, 'part-session': 10, 'therefore': 215, 'possibility': 3, 'agreement': 11, 'outside': 11, 'english': 1, 'island': 6, 'ladies': 15, 'six': 8, 'reconsider': 27, 'make': 24, 'imperative': 15, 'farmer': 15, 'include': 10, 'markets': 5, 'house': 30, 'certainty': 3, 'amendments': 15, 'allow': 15, 'vice-president': 14, 'wanted': 30, 'austrians': 26, 'authorities': 30, 'expertise': 15, 'remembered': 7, 'french': 15, 'environmental': 5, 'european': 8, 'afraid': 4, 'madam': 3, 'endorse': 15, 'particular': 15, 'members': 29, 'believe': 24, 'though': 15, 'commissioners': 6, 'politicians': 15, 'consider': 30, 'examine': 29, 'political': 5, 'conceivably': 7, 'precisely': 6, 'part': 11, 'igc': 15, 'orwellian': 6, 'matter': 11, 'questions': 45, 'complaint': 7, 'decision': 30, 'initiative': 11, 'carried': 5, '1': 13, 'absolutely': 11, 'january': 12, 'redraw': 5, 'actually': 15, 'colleagues': 15, 'done': 15, 'competent': 15, 'activities': 15, 'still': 22, 'finally': 38, 'obtained': 15, 'states': 15, 'use': 24, 'tax': 15, 'wager': 9, 'resort': 10, 'exercise': 6, 'family': 15, 'order': 4, 'given': 7, 'authorisations': 6, 'guarantee': 15, 'help': 15, 'byrne': 15, 'influence': 7, 'present': 10, 'bring': 14, 'extend': 15, 'bear': 10, 'again': 45, 'opportunity': 30, 'draft': 3, 'unacceptable': 15, 'instruments': 15, 'amendment': 30, 'request': 37, 'waiting': 11, 'specific': 5, 'spirit': 5, 'stopped': 45, 'once': 45, 'gentlemen': 6, 'wurtz': 26, 'conclusion': 25, 'ensures': 13, 'show': 12, 'important': 10, 'rural': 11, 'seek': 10, 'new': 22, 'values': 15, 'acting': 12, 'derogations': 6, '27': 9, 'kind': 30}

    # plot_senses(sense_dict, "Sense Vectors")
    # plot_senses(ctxt_dict, "Context Vectors")
    # plot_senses(my_dict, "My Cluster Centers")

    # plot_contexts("ask", ask_ctxt_words_sense_0, ask_ctxt_words_sense_1, "MSSG-tokenized_en-5-2-2")
    # plot_contexts(sense_dict, ctxt_dict, "ASK with contexts", m_common_0, m_common_1)

    # plot_spearmans("LOG_ENR")
    # plot_spearmans("LOG_SG")
    pass
