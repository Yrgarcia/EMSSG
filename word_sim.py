# -*- coding: utf-8 -*-
"""
Evaluation for trained word embeddings with multiple senses.
@author: Yoalli García
"""
import codecs
import numpy as np
import scipy.spatial.distance
from scipy.stats import spearmanr


class MSEmbeddings:
    def __init__(self, emb_file, sense_files):
        self.w2emb = extract_embs_from_file(emb_file)
        self.sense_dict = self.create_sense_dict(sense_files)
        self.k_senses = len(sense_files)
        self.stopwords = ['a',
                          'ain',
                          'all',
                          "also",
                          'am',
                          'an',
                          'and',
                          'any',
                          'are',
                          'aren',
                          "aren't",
                          "as",
                          'be',
                          'because',
                          'been',
                          'being',
                          'both',
                          'can',
                          'couldn',
                          "couldn't",
                          'd',
                          'did',
                          'didn',
                          "didn't",
                          'do',
                          'does',
                          'doesn',
                          "doesn't",
                          'doing',
                          'don',
                          "don't",
                          'each',
                          'few',
                          'further',
                          'had',
                          'hadn',
                          "hadn't",
                          'has',
                          'hasn',
                          "hasn't",
                          'have',
                          'haven',
                          "haven't",
                          'having',
                          'he',
                          'her',
                          'here',
                          'hers',
                          'herself',
                          'him',
                          'himself',
                          'his',
                          'how',
                          'i',
                          'if',
                          'is',
                          'isn',
                          "isn't",
                          'it',
                          "it's",
                          'its',
                          'itself',
                          'just',
                          'll',
                          'm',
                          'ma',
                          'me',
                          "might",
                          'mightn',
                          "mightn't",
                          'more',
                          'most',
                          "must",
                          'mustn',
                          "mustn't",
                          'my',
                          'myself',
                          "need",
                          'needn',
                          "needn't",
                          'no',
                          'nor',
                          'not',
                          'o',
                          'only',
                          'or',
                          'other',
                          'our',
                          'ours',
                          'ourselves',
                          'own',
                          're',
                          's',
                          'same',
                          "shall",
                          'shan',
                          "shan't",
                          'she',
                          "she's",
                          'should',
                          "should've",
                          'shouldn',
                          "shouldn't",
                          'so',
                          'some',
                          'such',
                          't',
                          'than',
                          'that',
                          "that'll",
                          'the',
                          'their',
                          'theirs',
                          'them',
                          'themselves',
                          'then',
                          'there',
                          'these',
                          'they',
                          'this',
                          'those',
                          'too',
                          "us",
                          've',
                          'very',
                          'was',
                          'wasn',
                          "wasn't",
                          'we',
                          'were',
                          'weren',
                          "weren't",
                          'what',
                          "when",
                          'where',
                          'which',
                          'who',
                          'whom',
                          'why',
                          'will',
                          'won',
                          "won't",
                          "would",
                          'wouldn',
                          "wouldn't",
                          'y',
                          'you',
                          "you'd",
                          "you'll",
                          "you're",
                          "you've",
                          'your',
                          'yours',
                          'yourself',
                          'yourselves']

    def create_sense_dict(self, sense_files):
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

    def get_most_probable_sense(self, w, c):
        sim_wc = 2.0
        sense = "NONE"
        for k in range(self.k_senses):
            sim_wc_new = scipy.spatial.distance.cosine(self.sense_dict[w][k], c)
            if sim_wc_new < sim_wc:
                sim_wc = sim_wc_new
                sense = k
        return sense

    def get_probability_of_sense(self, w, c):
        sim_wc = 2.0
        for k in range(self.k_senses):
            sim_wc_new = scipy.spatial.distance.cosine(self.sense_dict[w][k], c)
            if sim_wc_new < sim_wc:
                sim_wc = sim_wc_new
        return sim_wc

    def avg_sim(self, w, w_):
        # computes  the  average  similarity  over  all  embeddings  for  each  word,
        # ignoring  information  from the context
        similarity = 0.0
        factor = 1.0 / (self.k_senses ** 2)
        for i in range(self.k_senses):
            for j in range(self.k_senses):
                similarity += np.dot(self.sense_dict[w][i], self.sense_dict[w_][j]) / (np.linalg.norm(self.sense_dict[w][i]) * np.linalg.norm(self.sense_dict[w_][j]))
        return factor * similarity

    def avg_sim_c(self, w, w_, c, c_):
        # computes  the  average  similarity  over  all  embeddings  for  each  word,
        # including  information  from the context
        similarity = 0.0
        probability_w = self.get_probability_of_sense(w, c)
        probability_w_ = self.get_probability_of_sense(w_, c_)
        for i in range(self.k_senses):
            for j in range(self.k_senses):
                similarity += probability_w * probability_w_ * np.dot(self.sense_dict[w][i], self.sense_dict[w_][j]) / (np.linalg.norm(self.sense_dict[w][i]) * np.linalg.norm(self.sense_dict[w_][j]))
        return similarity

    def global_sim(self, w, w_):
        # Global context vector of word, ignoring senses
        similarity = np.dot(self.w2emb[w], self.w2emb[w_]) / (np.linalg.norm(self.w2emb[w]) * np.linalg.norm(self.w2emb[w_]))
        return similarity

    def local_sim(self, w, w_, c, c_):
        # LocalSim: single sense selection for each word based on the context
        k = self.get_most_probable_sense(w, c)
        k_ = self.get_most_probable_sense(w_, c_)
        similarity = np.dot(self.sense_dict[w][k], self.sense_dict[w_][k_]) / (np.linalg.norm(self.sense_dict[w][k]) * np.linalg.norm(self.sense_dict[w_][k_]))
        return similarity

    def max_sim(self, w, w_):
        similarity = 0.0
        for i in range(self.k_senses):
            for j in range(self.k_senses):
                new_sim = np.dot(self.sense_dict[w][i], self.sense_dict[w_][j]) / (
                            np.linalg.norm(self.sense_dict[w][i]) * np.linalg.norm(self.sense_dict[w_][j]))
                if new_sim > similarity:
                    similarity = new_sim
        return similarity

    def eval_on_multiple(self, eval_file, sim_type="globalSim"):
        ws353_pairs = {}
        with open(eval_file) as ws353:
            lines = ws353.readlines()
        for line in lines:
            lineslist = line.split()
            tup = (lineslist[0], lineslist[1])
            ws353_pairs[tup] = lineslist[2]

        my_scores = []
        gold_scores = []
        not_found = 0
        for pairs, score in zip(ws353_pairs.keys(), ws353_pairs.values()):
            try:
                if sim_type == "globalSim":
                    # print("\nMy score " + str((self.global_sim(pairs[0], pairs[1])) * 10))
                    my_scores.append((self.global_sim(pairs[0], pairs[1])) * 10)
                elif sim_type == "maxSim":
                    my_scores.append((self.max_sim(pairs[0], pairs[1])) * 10)
                elif sim_type == "avgSim":
                    my_scores.append((self.avg_sim(pairs[0], pairs[1])) * 10)
                # print("\nScore for " + str(pairs) + ": " + str(score))
                # print("\n=================================================")
                gold_scores.append(score)
            except KeyError:
                # print("\nNot found")
                not_found += 1
        # print len(gold_scores)
        # print len(my_scores)
        print("Found pairs in WS-353: " + str(353 - not_found) + " of " + str(353))
        print("Spearman Correlation for " + sim_type + " on WS-353: " + str(spearmanr(my_scores, gold_scores)))
        print("________________________________________________________")

    def calculate_ctxt_vecs_for_scws(self, contexts):
        # The target word is surrounded by <b>...</b> in its context.
        context_vectors = []
        not_in_data = 0
        for context in contexts:
            is_context_word = True
            sum_of_vc, v_count = 0, 0
            sctxt = context.split()
            for i in range(len(sctxt)):
                if sctxt[i] == "<b>":
                    is_context_word = False
                elif sctxt[i] == "</b>":
                    is_context_word = True
                elif is_context_word and sctxt[i] not in self.stopwords:
                    try:
                        sum_of_vc = np.add(sum_of_vc, self.w2emb[sctxt[i]])
                        v_count += 1
                    except KeyError:
                        not_in_data += 1
            context_vectors.append(sum_of_vc / v_count)
        print("\n" + str(not_in_data) + " context words don't occur in source data.")
        return context_vectors

    def eval_on_scws(self, scws_file, sim_type):
        # format: <id>   <word1>   <POS of word1>   <word2>   <POS of word2>   <word1 in context>   <word2 in context>
        #  <average human rating>   <10 individual human ratings>
        pairs, context_word1, context_word2, avg_rating = [], [], [], []

        with open(scws_file) as scws:
            lines = scws.readlines()
        for line in lines:
            sline = line.split("\t")
            pair = (sline[1], sline[3])
            pairs.append(pair)
            context_word1.append(sline[5])
            context_word2.append(sline[6])
            avg_rating.append(sline[7])

        # calculate the context vector from each context provided for a pair of words
        context_vecs1 = self.calculate_ctxt_vecs_for_scws(context_word1)
        context_vecs2 = self.calculate_ctxt_vecs_for_scws(context_word2)

        my_scores = []
        gold_scores = []
        not_found = 0

        for pairs, score, context_vec1, context_vec2 in zip(pairs, avg_rating, context_vecs1, context_vecs2):
            try:
                if sim_type == "globalSim":
                    # print("\nMy score " + str((self.global_sim(pairs[0], pairs[1]))*10))
                    my_scores.append((self.global_sim(pairs[0], pairs[1])) * 10)
                elif sim_type == "maxSim":
                    my_scores.append((self.max_sim(pairs[0], pairs[1])) * 10)
                elif sim_type == "avgSim":
                    # print("\nMy score " + str((self.avg_sim(pairs[0], pairs[1])) * 10))
                    my_scores.append((self.avg_sim(pairs[0], pairs[1])) * 10)
                elif sim_type == "avgSimC":
                    # print("\nMy score " + str((self.avg_sim_c(pairs[0], pairs[1], context_vec1, context_vec2)) * 10))
                    my_scores.append((self.avg_sim_c(pairs[0], pairs[1], context_vec1, context_vec2)) * 10)
                elif sim_type == "localSim":
                    # print("\nMy score " + str((self.local_sim(pairs[0], pairs[1], context_vec1, context_vec2)) * 10))
                    my_scores.append((self.local_sim(pairs[0], pairs[1], context_vec1, context_vec2)) * 10)

                # print("\nScore for " + str(pairs) + ": " + str(score) + "\nMy score: " + str(my_scores[-1]))
                # print("\n=================================================")
                gold_scores.append(score)

            except KeyError:
                not_found += 1
        print("\nFound: " + str(len(lines) - not_found) + " of " + str(len(lines)))
        my_scores = list(map(float, my_scores))
        # print(my_scores)
        gold_scores = list(map(float, gold_scores))
        spear = spearmanr(my_scores, gold_scores)
        print("Spearman Correlation for " + sim_type + " on SCWS: "  + str(spear))
        print("________________________________________________________")
        spearman_corr = spear[0]
        return spearman_corr

    def euclidean_distance(self, a, b):
        dist = np.linalg.norm(np.subtract(a, b))
        return dist

    def get_nearest_word(self, w):
        similarity = 0.0
        top_word = ""
        for word in self.w2emb.keys():
            if word == w:
                continue
            new_sim = self.euclidean_distance(self.w2emb[w], self.w2emb[word])
            if new_sim > similarity:
                similarity = new_sim
                top_word = word
        print("Nearest word to '" + w + "' is '" + str(top_word)+ "'.")
        return top_word

    def get_n_nearest_words(self, w, topnum):
        sim_dict = {}
        top_words = []
        for word in self.w2emb.keys():
            if word == w:
                continue
            new_sim = self.euclidean_distance(self.w2emb[w], self.w2emb[word])
            sim_dict[new_sim] = word
        sims = [sim for sim in sim_dict.keys()]
        sims.sort()
        for si in sims[-topnum:]:
            top_words.append(sim_dict[si])
        top_words.reverse()
        print("Nearest words to '" + w + "' are '" + str(top_words)+ "'.")
        return top_words

    def get_n_nearest_words_for_sense(self, w, sense, topnum):
        sim_dict = {}
        top_words = []

        for word in self.w2emb.keys():
            if word == w:
                continue
            new_sim = self.euclidean_distance(self.sense_dict[w][sense], self.w2emb[word])
            sim_dict[new_sim] = word
        sims = [sim for sim in sim_dict.keys()]
        sims.sort()
        for si in sims[-topnum:]:
            top_words.append(sim_dict[si])
        top_words.reverse()
        print("Nearest words to '" + w + "' are '" + str(top_words)+ "'.")
        return top_words

    def get_nearest_word_for_sense(self, w, sense):
        similarity = 0.0
        top_word = ""
        for word, emb in zip(self.w2emb.keys(), self.w2emb.values()):
            new_sim = self.euclidean_distance(self.sense_dict[w][sense], self.w2emb[word])
            if new_sim > similarity:
                similarity = new_sim
                top_word = word
        print("Nearest word to " + w + " is " + str(top_word)+ ".")
        return top_word

    def get_all_nearest(self, w):
        self.get_nearest_word(w)
        self.get_n_nearest_words(w, 20)
        self.get_nearest_word_for_sense(w, 0)
        self.get_nearest_word_for_sense(w, 1)
        self.get_n_nearest_words_for_sense(w, 0, 10)
        self.get_n_nearest_words_for_sense(w, 1, 10)


def extract_embs_from_file(filename):
    # get embeddings from a file
    # source format should be one embedding per line "<word> 0.1 0.2 0.3 ..."
    w2emb = {}
    with codecs.open(filename, "r", "utf-8") as f:
        lines = f.readlines()
    for line in lines:
        temp = line.split(" ")
        w2emb[temp[0]] = np.array([float(x) for x in temp[1:]])
    return w2emb


def evaluate(embedding_file, sim_type, sense_files=[]):
    scws_f = "SCWS/ratings.txt"
    print("SENSE_FILES:" + str(sense_files))
    emb = MSEmbeddings(embedding_file, sense_files)
    spearman_corr = emb.eval_on_scws(scws_f, sim_type)
    return spearman_corr


if __name__ == '__main__':
    # evaluate("EMSSG-tokenized_en-7-50-2", sim_type="globalSim", sense_files=["enr_SENSES_0", "enr_SENSES_1"])
    emb = MSEmbeddings("MSSG-TEST_en-5-2-2", ["not_enr_SENSES_0", "not_enr_SENSES_1"])
    emb.get_all_nearest("apple")
    # emb.get_nearest_word_for_sense("forest", 0)
    # emb.get_n_nearest_words_for_sense("forest", 0, 10)
    # emb.get_n_nearest_words_for_sense("forest", 1, 10)
    # emb.eval_on_multiple("WS-353/combined.tab", "globalSim")
    # evaluate("GENSIM_embs", sim_type="globalSim")
    # emb = MSEmbeddings(emb_file="RUN2LOG/MSSG-tokenized_en-7-100-2", sense_files=["RUN2LOG/not_enr_SENSES_0", "RUN2LOG/not_enr_SENSES_1"])
    # emb.eval_on_multiple("WS-353/combined.tab", "globalSim")
    # emb.eval_on_multiple("WS-353/combined.tab", "avgSim")
    # emb.eval_on_scws("SCWS/ratings.txt", "avgSim")
    pass
