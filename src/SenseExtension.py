# -*- coding: utf-8 -*-

"""
Extension to train MLP with additional sense vectors per word.
@author: Yoalli García
"""
import codecs
from scipy.spatial.distance import cosine
import numpy as np


class SenseExtension:
    def __init__(self, emb_file, sense_files, sentences, dim=100):
        self.global_w2vec = {}  # GLOBAL vectors for every global word
        self.w2emb, self.w2Idx = {}, {}  # COMBO: word to sense+global vecs; w2Idx: word to word index
        self.w2sense = {}  # RENAMED sense-words and sense vectors
        self.sense_words = []  # all words that have senses
        self.sense_dict = {}  # k sense vectors per word (SINGLE WORD entry)

        self.stopwords = {'a',
                     'ain',
                     'all',
                     'am',
                     'an',
                     'and',
                     'any',
                     'are',
                     'aren',
                     "aren't",
                     'be',
                     'because',
                     'been',
                     'being',
                     'both',
                     'but',
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
                     'down',
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
                     'mightn',
                     "mightn't",
                     'more',
                     'most',
                     'mustn',
                     "mustn't",
                     'my',
                     'myself',
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
                     'to',
                     'too',
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
                     'where',
                     'which',
                     'who',
                     'whom',
                     'why',
                     'will',
                     'won',
                     "won't",
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
                     'yourselves'}  # stopword set
        self.k_senses = len(sense_files)  # number of senses
        self.emb_file = emb_file  # name of file with pretrained global embeddings
        self.sense_files = sense_files  # list of names of sense vector files
        self.sentences = sentences  # list of lists of words of sentences from STREUSLE corpus

        self.extract_global_embs()
        self.get_all_embeddings(dim=dim)
        self.extract_sense_embs()
        self.create_sense_dict()

    def create_sense_dict(self):
        # load all embedding files embeddings (sense_files)
        for i in range(len(self.sense_files)):
            with open(self.sense_files[i]) as sf:
                lines = sf.readlines()
            for line in lines:
                temp_dict = {}
                line_list = line.split(" ")
                if i == 0:
                    temp_dict[i] = np.array([float(x) for x in line_list[1:]])
                    self.sense_dict[line_list[0]] = temp_dict
                else:
                    self.sense_dict[line_list[0]][i] = np.array([float(x) for x in line_list[1:]])
        self.sense_words = set(self.sense_dict.keys())

    def extract_sense_embs(self):
        # load sense embeddings (from sense_files)
        for i in range(len(self.sense_files)):
            with open(self.sense_files[i]) as sf:
                lines = sf.readlines()
            for line in lines:
                line_list = line.split(" ")
                self.w2sense[line_list[0] + str(i)] = np.array([float(x) for x in line_list[1:]])

    def extract_global_embs(self):
        # get embeddings from a file
        # source format should be one embedding per line "<word> 0.1 0.2 0.3 ..."
        with codecs.open(self.emb_file, "r", "utf-8") as f:
            lines = f.readlines()
        for line in lines:
            temp = line.split(" ")
            self.global_w2vec[temp[0]] = np.array([float(x) for x in temp[1:]])

    def get_all_embeddings(self, dim=100):
        # return dictionary containing all embeddings including sense embs and global embs
        for g_key in self.global_w2vec.keys():
            if g_key + "0" in self.w2sense.keys():
                self.w2emb[g_key + "0"] = self.w2sense[g_key + "0"]
            if g_key + "1" in self.w2sense.keys():
                self.w2emb[g_key + "1"] = self.w2sense[g_key + "1"]
            else:
                self.w2emb[g_key] = self.global_w2vec[g_key]
        idx = 0
        self.w2emb["PADDING"] = np.zeros(shape=dim)
        for word in self.w2emb.keys():
            self.w2Idx[word] = idx
            idx += 1

    def get_most_probable_sense(self, w, c):
            sim_wc = 2.0
            sense = "NONE"
            for k in range(self.k_senses):
                sim_wc_new = cosine(self.sense_dict[w][k], c)
                if sim_wc_new < sim_wc:
                    sim_wc = sim_wc_new
                    sense = k
            return sense

    def get_context(self, window, position, sentence):
        # get all context words from the word's context in window range
        current_window = np.random.randint(low=3, high=window + 1)
        context_start = max(position - current_window, 0)
        context_end = min(position + current_window + 1, len(sentence))
        context = sentence[context_start:position] + sentence[position + 1:context_end]
        return context, context_start, context_end

    def remove_stopwords_from_ctxt(self, window, context_, position, sentence, context_start, context_end):
        # Remove stop words from context:
        context = [w for w in context_ if w[1] not in self.stopwords]
        window_ = window
        while not context:
            if context_start == 0 and context_end == len(sentence):
                # if the context is already at its maximum and only filled with stopwords
                return context
                print("CONTEXT EMPTY")
            window_ += 1
            context_, context_start, context_end = self.get_context(window, position, sentence)
            context = [w for w in context_ if w[1] not in self.stopwords]
        return context

    def integrate_senses_to_data(self):
        # calculate context vector for each word in STREUSLE data that has a sense entry in sense_dict
        # [ [['Only', 'only', 'RB', '2', 'NMOD', '_', 7],[..],..], [[..],..], ...]
        not_in_data = 0
        words_with_sense_but_without_ctxt = 0
        window = 7
        new_sentences = []
        for sentence in self.sentences:
            new_sentence = []
            for i in range(len(sentence)):
                word = sentence[i][1]
                if word in self.sense_words:
                    # if word has senses, calculate context vector -> derive respective sense -> save back to list
                    # get context:
                    context_, cs, ce = self.get_context(window, i, sentence)
                    context = self.remove_stopwords_from_ctxt(window, context_, i, sentence, cs, ce)
                    sum_of_vc, v_count = 0, 0
                    for ctxt in context:
                        ctxt_word = ctxt[1]
                        try:
                            # exception for words that don't have an entry in embedding file
                            sum_of_vc = np.add(sum_of_vc, self.global_w2vec[ctxt_word])
                            v_count += 1
                        except KeyError:
                            not_in_data += 1
                    try:
                        ctxt_vec = sum_of_vc / v_count
                        # get sense k:
                        k = self.get_most_probable_sense(word, ctxt_vec)
                        # save back to list with respective sense-appendix
                        # ['server', 'server', 'NN', '6', 'SBJ', '_', 7] -> ['serverK', 'serverK', 'NN', '6', 'SBJ', '_', 7]
                        wordlist = [sentence[i][0] + str(k), sentence[i][1] + str(k)]
                        wordlist += sentence[i][-5:]
                        new_sentence.append(wordlist)
                    except ZeroDivisionError:
                        # it can happen that there's no embedding for any context word
                        # if so, simply skip as we cannot say which context/sense
                        words_with_sense_but_without_ctxt += 1
                        new_sentence.append(sentence[i])
                else:
                    # if word doesn't have different senses, simply append old word from data
                    new_sentence.append(sentence[i])
            new_sentences.append(new_sentence)
        print("\n" + str(not_in_data) + " context words don't occur in source data.")
        print("\n" + str(words_with_sense_but_without_ctxt) + " words' senses couldn't be computed because no context words were found.")
        return new_sentences


