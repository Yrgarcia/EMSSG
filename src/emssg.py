# -*- coding: utf-8 -*-
import math
import numpy as np
import os
import time
from src.word_sim import evaluate
from scipy.spatial.distance import cosine


class Corpus:
    # read in the tokenized corpus and extract all tokens
    def __init__(self, filename):
        file_pointer = open(filename, 'r')
        all_tokens = []
        self.notalnums = []
        for line in file_pointer:
            line_tokens = line.split()
            for token in line_tokens:
                token = token.lower()
                if len(token) < 2 and not token.isalnum():
                    self.notalnums.append(token)
                all_tokens.append(token)
        file_pointer.close()
        self.tokens = all_tokens

    def __getitem__(self, i):
        return self.tokens[i]

    def __len__(self):
        return len(self.tokens)

    def __iter__(self):
        return iter(self.tokens)


class Word:
    # word instance with string and count attribute
    def __init__(self, word):
        self.word = word
        self.count = 0


class Vocabulary:
    # build vocabulary out of the Corpus object, get N most common words, filter for rare and common words
    def __init__(self, corpus, min_count):
        self.words = []
        self.word_map = {}
        self.build_words(corpus)
        self.filter_for_rare_and_common(min_count)
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
        self.prepositions = ['of', 'with', 'at', 'from', 'into', 'during', 'including', 'until', 'against', 'among',
                             'throughout', 'despite', 'towards', 'upon', 'concerning', 'to', 'in', 'for', 'on', 'by',
                             'about', 'like', 'through', 'over', 'before', 'between', 'after', 'since', 'without',
                             'under', 'within', 'along', 'following', 'across', 'behind', 'beyond', 'plus', 'except',
                             'but', 'up', 'out', 'around', 'down', 'off', 'above', 'near']
        self.notalnums = corpus.notalnums

    def build_words(self, corpus):
        print("Building vocab...")
        words = []
        word_map = {}

        for token in corpus:
            if token not in word_map:
                word_map[token] = len(words)
                words.append(Word(token))
            words[word_map[token]].count += 1

        self.words = words
        self.word_map = word_map  # Mapping from each token to its index in vocab

    def __getitem__(self, i):
        return self.words[i]

    def __len__(self):
        return len(self.words)

    def __iter__(self):
        return iter(self.words)

    def __contains__(self, key):
        return key in self.word_map

    def indices(self, tokens):
        return [self.word_map[token] if token in self else self.word_map['UNKNOWN'] for token in tokens]

    def get_most_common(self, top_num, corpus):
        def get_eval_words(eval_data):
            words = []
            with open(eval_data) as scws:
                lines = scws.readlines()
            for line in lines:
                sline = line.split("\t")
                words.append(sline[1])
                words.append(sline[3])
            words = set(words)
            return words
        # TODO: remove when finished with testing
        eval_words = get_eval_words("SCWS/ratings.txt")

        word_count_pairs = []
        for word in self.words:
            # if word.word in eval_words and
            if word.word not in corpus.notalnums and word.word not in self.stopwords and word.word != "UNKNOWN" and word.word not in self.prepositions:
                temp = (word.word, word.count)
                word_count_pairs.append(temp)
        top_x = word_count_pairs[:top_num]
        top_x_words = [x[0] for x in top_x]
        # print("TOP %d words (within eval): %s " % (top_num, str(top_x_words)))
        return top_x_words

    def filter_for_rare_and_common(self, min_count):
        # Remove rare words and sort
        tmp = []
        tmp.append(Word('UNKNOWN'))
        unk_hash = 0

        count_unk = 0
        for token in self.words:
            if token.count < min_count:
                count_unk += 1
                tmp[unk_hash].count += token.count
            else:
                tmp.append(token)

        tmp.sort(key=lambda token : token.count, reverse=True)

        # Update word_map
        word_map = {}
        for i, token in enumerate(tmp):
            word_map[token.word] = i

        self.words = tmp
        self.word_map = word_map
        pass


class TableForNegativeSamples:
    # create the table for negative sampling
    def __init__(self, vocab):
        token2word = create_token2word(vocab)
        power = 0.75
        norm = sum([math.pow(t.count, power) for t in vocab if token2word[vocab.word_map[t.word]]])  # Normalizing constants

        table_size = 100000000
        # table_size = 100
        table = np.zeros(table_size, dtype=np.uint32)

        p = 0  # Cumulative probability
        i = 0
        for j, word in enumerate(vocab):
            if token2word[vocab.word_map[word.word]]:
                p += float(math.pow(word.count, power))/norm
                while i < table_size and float(i) / table_size < p:
                    table[i] = j
                    i += 1
        self.table = table

    def sample(self, count):
        indices = np.random.randint(low=0, high=len(self.table), size=count)
        return [self.table[i] for i in indices]


class Alignments:
    # extract alignments from file and reorganize
    def __init__(self, filename, corpus_en, corpus_es, trim=3000):
        self.alignments = self.convert_alignments(self.get_alignments(filename, trim), corpus_en, corpus_es)

    def get_alignments(self, filename, trim=3000):
        # extract alignments from the fast_align output file
        alignments = []
        with open(filename, "r") as al:
            lines = [next(al) for x in range(trim)]
        for line in lines:
            alis = []
            als = line.split(" ")
            for al in als:
                a = al.strip().split("-")
                for i in range(a.__len__()):
                    if a[i] != "":
                        a[i] = int(a[i])
                alis.append(a)
            alignments.append(alis)
        return alignments  # [[[0,1],[1,3],[2,1]],[[0,0],[1,1],[2,2]]]

    def convert_alignments(self, alignments, corpus_en, corpus_es):
        # convert alignments so that indices don't start from 0 for each sentence
        # -> one continuous corpus
        def get_lengths(corpus):
            # get the lengths of each sentence from corpus
            lengths = []
            sents = []
            with open(corpus) as en:
                lines = en.readlines()
            for line in lines:
                lineslist = line.strip().split()
                sents.append(lineslist)
            for sent in sents:
                lengths.append(len(sent))
            return lengths

        def add_one_by_one(l):
            # calculate sums of the position indices for later purpose
            new_l = []
            cumsum = 0
            for elt in l:
                cumsum += elt
                new_l.append(cumsum)
            return new_l[0:-1]

        def add_to_alignments(alignments, lengths_en, lengths_es):
            # add lengths of sentences
            new_alis = [x for x in alignments[0]]
            for i in range(1, len(alignments)):
                temp = []
                for k in range(len(alignments[i])):
                    if alignments[i][k] == ['']:
                        temp.append([""])
                    else:
                        temp.append([alignments[i][k][0] + lengths_en[i - 1], alignments[i][k][1] + lengths_es[i - 1]])
                new_alis += temp
            return new_alis

        lengths_es = get_lengths(corpus_es)
        lengths_en = get_lengths(corpus_en)
        new_l_en = add_one_by_one(lengths_en)
        new_l_es = add_one_by_one(lengths_es)
        new_als = add_to_alignments(alignments, new_l_en, new_l_es)
        return new_als


def reverse_alignments(alignment_file, corpus_en, corpus_es, trim=3000):
    # Test whether alignment conversion was successful
    converted_als = Alignments(alignment_file, corpus_en, corpus_es, trim=trim).alignments
    min_count = 5  # Min count for words to be used in the model, else UNKNOWN
    corpus = Corpus(corpus_en)
    corpus_ = Corpus(corpus_es)
    vocab = Vocabulary(corpus, min_count)
    vocab_ = Vocabulary(corpus_, min_count)
    tokens_ = vocab_.indices(corpus_)
    token2word = {}
    for key, val in zip(vocab.word_map.values(), vocab.word_map.keys()):
        token2word[key] = val
    token2word_ = {}
    for key, val in zip(vocab_.word_map.values(), vocab_.word_map.keys()):
        token2word_[key] = val
    new_corpus_ = []
    indexerrors = 0
    for al in converted_als:
        if al != [""]:
            try:
                #if token2word_[tokens_[al[1]]] not in [".", ",", "(", ")", "'"]:
                new_corpus_.append(token2word_[tokens_[al[1]]])
            except IndexError:
                indexerrors += 1
                new_corpus_.append(str(indexerrors))
    with open("1reversed_alignments.txt", "w") as ra:
        for word in new_corpus_:
            ra.write(word)
            ra.write(" ")
    ra.close()
    print("INDEXERRORS: " +str(indexerrors))


def sigmoid(z):
    # map z to [0:1]
    if z > 6:
        return 1.0
    elif z < -6:
        return 0.0
    else:
        return 1 / (1 + math.exp(-z))


def save(vocab, nn0, filename, token2word):
    # save embeddings to file
    # vocab can be either an instance of the vocab class or a dictionary mapping words to their embeddings
    a = {0: 0}
    file_pointer = open(filename, 'w')
    for token, vector in zip(vocab, nn0):
        if type(vocab) != type(a.keys()):
            word = token.word.replace(' ', '_')
        else:
            word = token
        vector_str = ' '.join([str(s) for s in vector])
        if type(vocab) != type(a.keys()):
            if token2word[vocab.word_map[word]]:
                file_pointer.write('%s %s\n' % (word, vector_str))
        else:
            file_pointer.write('%s %s\n' % (word, vector_str))
    file_pointer.close()


def create_token2word(vocab, use_prepositions=False):
    # create a dictionary that maps each token to its word string, with stop words mapping to False
    token2word = {}
    if use_prepositions:
        stop = vocab.stopwords + list(set(vocab.notalnums))
    else:
        stop = vocab.stopwords + list(set(vocab.notalnums)) + vocab.prepositions
    for key, val in zip(vocab.word_map.values(), vocab.word_map.keys()):
        if val not in stop:
            token2word[key] = val
        else:
            # add False value to filter out stop words while training
            token2word[key] = False
    return token2word


def calc_my_update(current_vector_count, curr_my, average):
    # the update is calculated with the average context vector,
    # weighed according to the number of vectors already in the cluster
    zaehler = np.add((current_vector_count * curr_my), average)
    nenner = math.pow(current_vector_count + 1, -1)
    product = zaehler * nenner
    return product


def get_nearest_my(old_my, average, maximum, k):
    # returns the nearest context cluster sense for the current context
    new_max = cosine(old_my, average)
    max_k = 0
    if new_max <= maximum:
        maximum = new_max
        max_k = k
    return maximum, max_k


def backpropagate_n_negsampling(dim, classifiers, v_c, context_word, v_s_wk, s_t, alpha):
    # backpropagation including negative sampling
    neu1e = np.zeros(dim)
    for target, label in classifiers:  # label: 0 if word is not token, 1 if it is
        z = np.dot(v_c[context_word], v_s_wk[target][s_t])
        p = sigmoid(z)
        g = alpha * (label - p)  # g positive for token, negative for targets
        neu1e += g * v_s_wk[target][s_t]  # Error to backpropagate to v_c
        v_s_wk[target][s_t] += g * v_c[context_word]  # Update v_s_wk(w,s_t) with v_c(c)
    # Update v_c(c) with v_s_wk:
    v_c[context_word] += neu1e
    return v_s_wk, v_c


def gradient_update(dim, token, table, k_negative_sampling, v_c, context, v_s_wk, s_t, alpha):
    for context_word in context:
        # perform gradient updates:
        # for every word in the token's context: sample k negative examples
        # Init neu1e with zeros
        classifiers = [(token, 1)] + [(target, 0) for target in table.sample(k_negative_sampling)]
        v_s_wk, v_c = backpropagate_n_negsampling(dim, classifiers, v_c, context_word, v_s_wk, s_t, alpha)
    return v_c, v_s_wk


def log_spearman(spearman_corr, filename):
    # write the Spearman correlation from each iteration to a file
    with open(filename, "a") as sl:
        sl.write(str(spearman_corr)+"\n")
        sl.close()


def get_context(window, token_idx, tokens, rand=True):
    # return the context of current token
    if rand:
        current_window = np.random.randint(low=3, high=window + 1)
    else:
        current_window = window
    context_start = max(token_idx - current_window, 0)
    context_end = min(token_idx + current_window + 1, len(tokens))
    context = tokens[context_start:token_idx] + tokens[token_idx + 1:context_end]
    return context, context_start, context_end


def save_eval_log_senses(words_for_sense_training, vocab, num_of_senses, embedding_file, token2word, v_s_wk, enr):
    # Save sense embeddings to files:
    all_senses = [0] * num_of_senses
    for k in range(num_of_senses):
        all_senses[k] = {}

    for i in range(len(vocab.words)):
        if vocab.words[i].word in words_for_sense_training:  # PREPOSITION CHANGE
            for k in range(num_of_senses):
                all_senses[k][vocab.words[i].word] = v_s_wk[vocab.word_map[vocab.words[i].word]][k]
            # senses1[vocab.words[i].word] = v_s_wk[vocab.word_map[vocab.words[i].word]][1]
    sense_files = []
    for k in range(num_of_senses):
        sense_files.append("%sSENSES_%d" % (enr, k))
        save(all_senses[k].keys(), all_senses[k].values(), "%sSENSES_%d" % (enr, k), token2word)
    # save(senses1.keys(), senses1.values(), "%sSENSES_1" % enr, token2word)
    # Evaluate sense embeddings:
    spearman = evaluate(embedding_file, "localSim", sense_files=sense_files)
    log_spearman(spearman, "../Spearman_local_%ssenses" % enr)
    # save best senses to BEST_enr_SENSES_* or BEST_not_enr_SENSES_*
    # if spearman > old_spearman:
    #    old_spearman = spearman
    #    for k in range(num_of_senses):
    #        save(senses[k].keys(), senses[k].values(), "BEST_" + enr + "SENSES_" + str(k))
    print("\n===========================================================")


def save_eval_log_global(vocab, v_c, embedding_file, token2word, enr):
    # Save context embeddings to file:
    save(vocab, v_c, embedding_file, token2word)
    # Evaluate context embeddings:
    sp = evaluate(embedding_file, "globalSim")
    log_spearman(sp, "../Spearman_global_%scontext_embs" % enr)
    # save best embeddings to BEST_MSSG_embs
    # if sp > old_sp_ctxt:
    #   old_sp_ctxt = sp
    #   save(vocab, v_c, "BEST_" + embedding_file)


def train_mssg(corpus_en, corpus_es, epochs, dim, enriched, use_prepositions, window, verbose, alpha_0, topnum, num_of_senses, min_count):
    # main function for training the MSSG and EMSSG models
    k_negative_sampling = 5  # Number of negative samples
    # Initial learning rate:
    alpha = alpha_0
    embedding_file = 'MSSG-%s-%d-%d-%d' % (corpus_en, window, dim, num_of_senses)
    corpus = Corpus(corpus_en)
    vocab = Vocabulary(corpus, min_count)
    table = TableForNegativeSamples(vocab)
    tokens = vocab.indices(corpus)
    token2word = create_token2word(vocab, use_prepositions)
    most_common_words = vocab.get_most_common(topnum, corpus)
    vector_count = {}  # for counting vectors in iteration
    print("Training: %s-%d-%d-%d" % (corpus_en, window, dim, num_of_senses))
    # Initialize network:
    my_wk = np.zeros(shape=(len(vocab), num_of_senses, dim))
    np.random.seed(3)
    v_s_wk = np.random.uniform(low=-0.5 / dim, high=0.5 / dim, size=(len(vocab), num_of_senses, dim))
    np.random.seed(7)
    v_c = np.random.uniform(low=-0.5 / dim, high=0.5 / dim, size=(len(vocab), dim))
    enr = "not_enr_"
    # if senses are trained for prepositions, prepositions should be excluded from the stop word filter
    if use_prepositions:
        words_for_sense_training = vocab.prepositions
        is_not_stopword = create_token2word(vocab, use_prepositions=False)
    else:
        words_for_sense_training = most_common_words
        is_not_stopword = token2word
    for word in words_for_sense_training:       # prepare vector count for training my
        temp_dict = {}
        for k in range(num_of_senses):
            temp_dict[k] = 0
        vector_count[word] = temp_dict

    if enriched:
        embedding_file = 'EMSSG-%s-%d-%d-%d' % (corpus_en, window, dim, num_of_senses) # change name of output file
        converted_als = Alignments("aligned_file", corpus_en, corpus_es).alignments  # load the converted alignments
        corpus_ = Corpus(corpus_es)  # extract the second language corpus
        vocab_ = Vocabulary(corpus_, min_count)  # extract vocabulary from corpus
        tokens_ = vocab_.indices(corpus_)  # get all tokens
        enriched_contexts = [[] for _ in range(len(tokens))]  # map a list of aligned words to each token
        for al in converted_als:
            if al != [""]:
                enriched_contexts[al[0]].append(tokens_[al[1]] + len(vocab))
        # ENR: v_c_ for enriched context vectors:
        np.random.seed(7)
        v_c = np.random.uniform(low=-0.5 / dim, high=0.5 / dim, size=(len(vocab)+len(vocab_), dim))
        enr = "enr_"
    mys = []  # for saving mys to file
    # Fill mys and a temporary dict for later use (vector_count)
    for k in range(num_of_senses):
        mys.append(np.zeros(shape=(len(vocab), dim)))
    # Start training:
    for epoch in range(epochs):
        print(enr + "EPOCH: " + str(epoch))
        for token_idx, token in enumerate(tokens):
            if token2word[token]:
                # Get sg context from context window:
                context_, context_start, context_end = get_context(window, token_idx, tokens)
                # Remove stop words from context and refill while empty:
                context = [tok for tok in context_ if is_not_stopword[tok]]
                window_ = int(len(context_)/2)
                while not context and window_ < 10:
                    # if context only contains stop words, expand context
                    window_ += 1
                    context_, context_start, context_end = get_context(window_, token_idx, tokens, rand=False)
                    context = [tok for tok in context_ if token2word[tok]]
                if enriched:
                    # retrieve aligned words from enriched corpus and unify with skip-gram context
                    enriched_context = enriched_contexts[token_idx]
                    context += enriched_context
# ###################################### SENSE TRAINING #################################################
                s_t = 0  # if there's no sense training for token, use sense=0 as default
                if token2word[token] in words_for_sense_training:  # PREPOSITION CHANGE
                    # ########### get sum of all context vectors #################################
                    sum_of_vc = np.zeros(dim)
                    for context_word in context:
                        sum_of_vc = np.add(sum_of_vc, v_c[context_word])
                    # ########### calculate average of context words' vectors ####################
                    try:
                        average = sum_of_vc * math.pow(len(context), -1)
                    except ValueError:
                        # exception if context is empty/full of stop words
                        print(str(token2word[tokens[token_idx-14]]) + str(token2word[tokens[token_idx-13]]) + str(token2word[tokens[token_idx-12]]) + str(token2word[tokens[token_idx-11]]) + token2word[token])
                    # ########### get nearest sense k (s_t) from sim(my(w_t,k), sum_of_vc) #######
                    maximum = 2.0
                    for k in range(num_of_senses):
                        old_my = my_wk[token][k]  # get old cluster centre
                        # look for nearest my(token, k), that is: max(my(token,k) * average)
                        maximum, max_k = get_nearest_my(old_my, average, maximum, k)
                    s_t = max_k  # nearest sense s_t

                    # ############### update cluster centre my_wk ################################
                    current_vector_count = vector_count[token2word[token]][s_t]
                    curr_my = my_wk[token][s_t]
                    product = calc_my_update(current_vector_count, curr_my, average)
                    my_wk[token][s_t] = product
                    mys[s_t][token] = product
                    vector_count[token2word[token]][s_t] += 1
# ###################################### GRADIENT UPDATE #################################################
                v_c, v_s_wk = gradient_update(dim, token, table, k_negative_sampling, v_c, context, v_s_wk, s_t, alpha)
        # update learning rate
        alpha = 0.95**epoch * alpha_0
        if verbose:
            print(vector_count)
        # plot("ask", count_ctxt_words_for_principle_0, count_ctxt_words_for_principle_1, embedding_file)
        save_eval_log_global(vocab, v_c, embedding_file, token2word, enr)
        save_eval_log_senses(words_for_sense_training, vocab, num_of_senses, embedding_file, token2word, v_s_wk, enr)
    # return v_s_wk, v_c, my_wk  # ENR: return v_c_
    return embedding_file


def execute_emssg_or_mssg(config):
    # extract parameters from config and execute training processes
    os.chdir("./src/")
    params = config["EMSSG"]
    english_corpus = "tokenized_en"
    foreign_corpus = "tokenized_%s" % params["language"]
    epochs = params["epochs"]
    dim = params["dimension"]
    enriched = params["enriched"]
    window = params["window"]
    use_prepositions = params["use prepositions"]
    verbose = params["print cluster counts"]
    alpha = params["learning rate"]
    topnum = params["most common"]
    senses = params["senses"]
    min_count = params["min_count"]

    start = time.time()
    train_mssg(english_corpus, foreign_corpus, epochs, dim, enriched, use_prepositions, window, verbose, alpha, topnum, senses, min_count)
    end = time.time()
    print("\nIt took: " + str(round((end-start)/60)) + "min to run.")


if __name__ == '__main__':
    pass
