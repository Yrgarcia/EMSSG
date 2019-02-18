import math
import sys
import numpy as np
import matplotlib.pyplot as plt
import time
from preprocessData import PreprocessData
from word_sim import evaluate
from word_sim import extract_embs_from_file
from scipy.spatial.distance import cosine


class Ngram:
    def __init__(self, tokens):
        self.tokens = tokens
        self.count = 0
        self.score = 0.0

    def set_score(self, score):
        self.score = score

    def get_string(self):
        return '_'.join(self.tokens)


class Corpus:
    def __init__(self, filename_list, word_phrase_passes, word_phrase_delta, word_phrase_threshold, word_phrase_filename, trim=3000):
        all_tokens = []
        self.notalnums = []

        for filename in filename_list:
            with open(filename, 'r') as myfile:
                lines = [next(myfile) for x in range(trim)]
            for line in lines:
                line_tokens = line.split()
                for token in line_tokens:
                    token = token.lower()

                    if len(token) < 2 and not token.isalnum():
                        self.notalnums.append(token)
                    all_tokens.append(token)
            #print("NOTALNUMS: " + str(set(notalnums)))

            #file_pointer.close()

        self.tokens = all_tokens

        #for x in range(1, word_phrase_passes + 1):
        #   self.build_ngrams(x, word_phrase_delta, word_phrase_threshold, word_phrase_filename)

        self.save_to_file(filename)

    def build_ngrams(self, x, word_phrase_delta, word_phrase_threshold, word_phrase_filename):

        ngrams = []
        ngram_map = {}

        token_count_map = {}
        for token in self.tokens:
            if token not in token_count_map:
                token_count_map[token] = 1
            else:
                token_count_map[token] += 1

        i = 0
        ngram_l = []
        for token in self.tokens:

            if len(ngram_l) == 2:
                ngram_l.pop(0)

            ngram_l.append(token)
            ngram_t = tuple(ngram_l)

            if ngram_t not in ngram_map:
                ngram_map[ngram_t] = len(ngrams)
                ngrams.append(Ngram(ngram_t))

            ngrams[ngram_map[ngram_t]].count += 1

            i += 1
            if i % 10000 == 0:
                sys.stdout.flush()
                sys.stdout.write("\rBuilding n-grams (%d pass): %d" % (x, i))
        sys.stdout.flush()
        print( "\rn-grams (%d pass) built: %d" % (x, i))
        filtered_ngrams_map = {}
        file_pointer = open(word_phrase_filename + ('-%d' % x), 'w')

        # http://papers.nips.cc/paper/5021-distributed-representations-of-words-and-phrases-and-their-compositionality.pdf
        i = 0
        for ngram in ngrams:
            product = 1
            for word_string in ngram.tokens:
                product *= token_count_map[word_string]
            ngram.set_score((float(ngram.count) - word_phrase_delta) / float(product))
            if ngram.score > word_phrase_threshold:
                filtered_ngrams_map[ngram.get_string()] = ngram
                file_pointer.write('%s %d\n' % (ngram.get_string(), ngram.count))
            i += 1
            if i % 10000 == 0:
                sys.stdout.flush()
                sys.stdout.write("\rScoring n-grams: %d" % i)

        sys.stdout.flush()
        print("\rScored n-grams: %d, filtered n-grams: %d" % (i, len(filtered_ngrams_map)))
        file_pointer.close()
        # Combining the tokens
        all_tokens = []
        i = 0
        while i < len(self.tokens):
            if i + 1 < len(self.tokens):
                ngram_l = []
                ngram_l.append(self.tokens[i])
                ngram_l.append(self.tokens[i+1])
                ngram_string = '_'.join(ngram_l)

                if len(ngram_l) == 2 and (ngram_string in filtered_ngrams_map):
                    ngram = filtered_ngrams_map[ngram_string]
                    all_tokens.append(ngram.get_string())
                    i += 2
                else:
                    all_tokens.append(self.tokens[i])
                    i += 1
            else:
                all_tokens.append(self.tokens[i])
                i += 1

        print( "Tokens combined")

        self.tokens = all_tokens

    def save_to_file(self, filename):

        i = 1

        filepointer = open('preprocessed-' + filename, 'w')
        line = ''
        for token in self.tokens:
            if i % 20 == 0:
                line += token
                filepointer.write('%s\n' % line)
                line = ''
            else:
                line += token + ' '

        #print("\rPreprocessed input file written")

        filepointer.close()

    def filter_for_stop_words(self):
        stopwords = {'a',
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
 'yourselves'}
        # TODO
        return 0

    def __getitem__(self, i):
        return self.tokens[i]

    def __len__(self):
        return len(self.tokens)

    def __iter__(self):
        return iter(self.tokens)


class Word:
    def __init__(self, word):
        self.word = word
        self.count = 0


class Vocabulary:
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
                          'to',
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
            if word.word in eval_words and word.word not in corpus.notalnums and word.word not in self.stopwords and word.word != "UNKNOWN" and word.word not in self.prepositions:
                temp = (word.word, word.count)
                word_count_pairs.append(temp)
        top_x = word_count_pairs[:top_num]
        top_x_words = [x[0] for x in top_x]
        print("TOP %d words (within eval): %s " % (top_num, str(top_x_words)))
        return top_x_words

    def get_most_common_prepositions(self, top_num):
        word_count_pairs = []
        for word in self.words:
            if word.word in self.prepositions:
                temp = (word.word, word.count)
                word_count_pairs.append(temp)
        top_x = word_count_pairs[:top_num]
        top_x_words = [x[0] for x in top_x]
        # print(top_x_words)
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
    def __init__(self, vocab):
        power = 0.75
        norm = sum([math.pow(t.count, power) for t in vocab]) # Normalizing constants

        table_size = 100000000
        table = np.zeros(table_size, dtype=np.uint32)

        p = 0 # Cumulative probability
        i = 0
        for j, word in enumerate(vocab):
            p += float(math.pow(word.count, power))/norm
            while i < table_size and float(i) / table_size < p:
                table[i] = j
                i += 1
        self.table = table

    def sample(self, count):
        indices = np.random.randint(low=0, high=len(self.table), size=count)
        return [self.table[i] for i in indices]


class Alignments:
    def __init__(self, filename, corpus_en, corpus_es, trim=3000):
        self.alignments = self.convert_alignments(self.get_alignments(filename, trim), corpus_en, corpus_es)
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
                          'to',
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


def sigmoid(z):
    if z > 6:
        return 1.0
    elif z < -6:
        return 0.0
    else:
        return 1 / (1 + math.exp(-z))


def save(vocab, nn0, filename):
    a = {0: 0}
    file_pointer = open(filename, 'w')
    for token, vector in zip(vocab, nn0):
        if type(vocab) != type(a.keys()):
            word = token.word.replace(' ', '_')
        else:
            word = token
        vector_str = ' '.join([str(s) for s in vector])
        file_pointer.write('%s %s\n' % (word, vector_str))
    file_pointer.close()


def get_prepositions(filename):
    prepositions = []
    with open(filename, "r") as prepos:
        lines = prepos.readlines()
        for p in lines:
            p = p.lower().strip()
            prepositions.append(p)
    return prepositions


def create_token2word(vocab):
    token2word = {}
    for key, val in zip(vocab.word_map.values(), vocab.word_map.keys()):
        token2word[key] = val
    return token2word


def calc_my_update(current_vector_count, curr_my, average):
    zaehler = np.add((current_vector_count * curr_my), average)
    nenner = math.pow(current_vector_count + 1, -1)
    product = zaehler * nenner
    return product


def get_nearest_my(old_my, average, maximum, k):
    if old_my.all() == 0.0:
        max_k = k
    else:
        new_max = cosine(old_my, average)
        max_k = 0
        if new_max <= maximum:
            maximum = new_max
            max_k = k
    return maximum, max_k


def backpropagate_n_negsampling(dim, classifiers, v_c, context_word, v_s_wk, s_t, alpha):
    neu1e = np.zeros(dim)
    for target, label in classifiers:  # label: 0 if word is not token, 1 if it is
        p = cosine(v_c[context_word], v_s_wk[target][s_t]) - 1.0
        g = (alpha * (p + label))  # g positive for token, negative for targets
        neu1e += g * v_s_wk[target][s_t]  # Error to backpropagate to v_s_wk
        v_s_wk[target][s_t] += g * v_c[context_word]  # Update v_s_wk(w,s_t) with v_c(c)
    # ##############  Update v_c(c) with v_s_wk: ##########################
    v_c[context_word] += neu1e
    return v_s_wk, v_c


def gradient_update(dim, token, table, k_negative_sampling, v_c, context, v_s_wk, s_t, alpha, senses, token2word, most_common_words):
    for context_word in context:
        # perform gradient updates:
        # for every word in the token's context: sample k negative examples
        # Init neu1e with zeros
        classifiers = [(token, 1)] + [(target, -1) for target in table.sample(k_negative_sampling)]
        v_s_wk, v_c = backpropagate_n_negsampling(dim, classifiers, v_c, context_word, v_s_wk, s_t, alpha)
        if token2word[token] in most_common_words:
            senses[s_t][token2word[token]] = v_s_wk[token][s_t]
    return v_c, senses, v_s_wk


def log_spearman(spearman_corr, filename):
    with open(filename, "a") as sl:
        sl.write(str(spearman_corr)+"\n")
        sl.close()


def get_context(window, token_idx, tokens):
    current_window = np.random.randint(low=3, high=window + 1)
    context_start = max(token_idx - current_window, 0)
    context_end = min(token_idx + current_window + 1, len(tokens))
    context = tokens[context_start:token_idx] + tokens[token_idx + 1:context_end]
    return context, context_start, context_end


def emssg(corpus_en, corpus_es=None, alignment_file=None, dim=100, epochs=10, enriched=False, trim=3000):
    num_of_senses = 2  # 2; number of senses
    window = 7  # Max window length: 5 for large set(excluding
    k_negative_sampling = 5  # Number of negative examples
    min_count = 3  # Min count for words to be used in the model, else UNKNOWN
    # Initial learning rate:
    alpha_0 = 0.01  # 0.01
    alpha = alpha_0
    embedding_file = 'MSSG-%s-%d-%d-%d' % (corpus_en, window, dim, num_of_senses)
    old_spearman = 0  # for best sense spearman
    old_sp_ctxt = 0  # for best global embedding spearman
    old_sp_my = 0  # for testing cluster centre
    word_phrase_passes = 3  # 3; Number of word phrase passes
    word_phrase_delta = 3   # 5; min count for word phrase formula
    word_phrase_threshold = 1e-4  # Threshold for word phrase creation
    corpus = Corpus([corpus_en], word_phrase_passes, word_phrase_delta, word_phrase_threshold, 'phrases-%s' % corpus_en, trim=trim)
    vocab = Vocabulary(corpus, min_count)
    table = TableForNegativeSamples(vocab)
    tokens = vocab.indices(corpus)
    token2word = create_token2word(vocab)
    # most_common_preps = vocab.get_most_common_prepositions(100)
    most_common_words = vocab.get_most_common(1000, corpus)
    stop_words = vocab.stopwords
    notalnums = vocab.notalnums
    stop_notal = notalnums + stop_words
    print("Training: %s-%d-%d-%d" % (corpus_en, window, dim, num_of_senses))
    # Initialize network:
    my_wk = np.zeros(shape=(len(vocab), num_of_senses, dim))
    np.random.seed(3)
    v_s_wk = np.random.uniform(low=-0.5 / dim, high=0.5 / dim, size=(len(vocab), num_of_senses, dim))
    np.random.seed(7)
    v_c = np.random.uniform(low=-0.5 / dim, high=0.5 / dim, size=(len(vocab), dim))
    enr = "not_enr_"
    if enriched:
        embedding_file = 'EMSSG-%s-%d-%d-%d' % (corpus_en, window, dim, num_of_senses)
        converted_als = Alignments(alignment_file, corpus_en, corpus_es, trim=trim).alignments
        corpus_ = Corpus([corpus_es], word_phrase_passes, word_phrase_delta, word_phrase_threshold, 'phrases-%s' % corpus_es)
        combined_corpus = Corpus([corpus_en, corpus_es], word_phrase_passes, word_phrase_delta, word_phrase_threshold, 'phrases-%s' % corpus_en)
        vocab_ = Vocabulary(corpus_, min_count)
        combined_vocab = Vocabulary(combined_corpus, min_count)
        tokens_ = vocab_.indices(corpus_)
        # ENR: v_c_ for enriched context vectors:
        np.random.seed(7)
        v_c = np.random.uniform(low=-0.5 / dim, high=0.5 / dim, size=(len(combined_vocab), dim))
        enr = "enr_"
    vector_count = {}  # for counting vectors in iteration
    temp_fill_dict_vc = {}  # temp dict for filling vector_count
    mys = []  # for saving mys to file
    t_dict_for_senses = {}
    for word in most_common_words:
        t_dict_for_senses[word] = np.zeros(shape=dim)
    senses = [t_dict_for_senses for i in range(num_of_senses)]  # for saving senses to file: senses[0] = vectors for each word for s0 =[{"word":[1.23,0.56,...]},{...}]
    # Fill mys, senses and a temporary dict for later use (vector_count)
    for k in range(num_of_senses):
        temp_fill_dict_vc[k] = 0
        mys.append(np.zeros(shape=(len(vocab), dim)))
        # senses.append(np.zeros(shape=(len(vocab), dim)))
    # Start training:
    for epoch in range(epochs):
        print(enr + "EPOCH: " + str(epoch))
        for token_idx, token in enumerate(tokens):
            if token2word[token] not in stop_notal:
                if epoch == 0:
                    vector_count[token2word[token]] = temp_fill_dict_vc
                # Get sg context from context window:
                context_, context_start, context_end = get_context(window, token_idx, tokens)
                # Remove stop words from context:
                context = [tok for tok in context_ if token2word[tok] not in stop_notal]
                window_ = window
                while not context:
                    window_ += 1
                    context_, context_start, context_end = get_context(window, token_idx, tokens)
                    context = [tok for tok in context_ if token2word[tok] not in stop_notal]
                # ENR: get enriched context and unify
                if enriched:
                    enriched_context = []
                    enriched_context_als_ = converted_als[context_start:token_idx] + converted_als[token_idx + 1:context_end]
                    # remove english stop words and aligned spanish word:
                    enriched_context_als = [tok for tok in enriched_context_als_ if token2word[tok[0]] not in stop_notal]
                    for als in enriched_context_als:
                        # go through retrieved alignments and get token IDs from corresponding aligned tokens
                        if als != [""]:
                            enriched_context.append(tokens_[als[1]])
                    context += enriched_context
# ###################################### SENSE TRAINING #################################################
                s_t = 0  # if there's no sense training for token, use sense=0 as default
                if token2word[token] in most_common_words:
                    # ########### get sum of all context vectors #################################
                    sum_of_vc = np.zeros(dim)
                    for context_word in context:
                        sum_of_vc = np.add(sum_of_vc, v_c[context_word])
                    # ########### calculate average of context words' vectors ####################
                    average = sum_of_vc * math.pow(len(context), -1)
                    # ########### get nearest sense k (s_t) from sim(my(w_t,k), sum_of_vc) #######
                    maximum = 1.0
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
                v_c, senses, v_s_wk = gradient_update(dim, token, table, k_negative_sampling, v_c, context, v_s_wk, s_t, alpha, senses, token2word, most_common_words)

        # update learning rate
        alpha = 0.95**epoch * alpha_0

        # Save context embeddings to file:
        save(vocab, v_c, embedding_file)
        # Evaluate context embeddings:
        sp = evaluate(embedding_file, "globalSim")
        log_spearman(sp, "LOG_%scontext_embs" % enr)
        # save best embeddings to BEST_MSSG_embs
        # if sp > old_sp_ctxt:
        #   old_sp_ctxt = sp
        #   save(vocab, v_c, "BEST_" + embedding_file)

        # # Save cluster centres to files
        # for k in range(num_of_senses):
        #     save(vocab, mys[k], enr + "MYS_" + str(k))
        # # Evaluate sense embeddings:
        # sp_my = evaluate("BEST_" + embedding_file, "localSim", enr=False, sense_files=["not_enr_MYS_0", "not_enr_MYS_1"])
        # log_spearman(sp_my, "LOG_" + enr + "MYS")
        # # save best senses to BEST_MYS_*
        # if sp_my > old_sp_my:
        #     old_sp_my = sp_my
        #     for k in range(num_of_senses):
        #         save(vocab, mys[k], "BEST_" + enr + "MYS_" + str(k))

        # Save sense embeddings to files
        for k in range(num_of_senses):
            save(senses[k].keys(), senses[k].values(), "%sSENSES_%s" % (enr, str(k)))
        # Evaluate sense embeddings:
        spearman = evaluate(embedding_file, "localSim", sense_files=["%sSENSES_0" % enr, "%sSENSES_1" % enr])
        log_spearman(spearman, "LOG_%ssenses" % enr)
        # save best senses to BEST_enr_SENSES_* or BEST_not_enr_SENSES_*
        # if spearman > old_spearman:
        #    old_spearman = spearman
        #    for k in range(num_of_senses):
        #        save(senses[k].keys(), senses[k].values(), "BEST_" + enr + "SENSES_" + str(k))
        print("\n===========================================================")
    # return v_s_wk, v_c, my_wk  # ENR: return v_c_
    return embedding_file


def normalize(v):
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm


def plot_senses(sense_dict):
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
    plt.legend(ncol=2)
    plt.show()


def reverse_alignments(alignment_file, corpus_en, corpus_es, trim=3000):
    # Test whether alignment conversion was successful
    converted_als = Alignments(alignment_file, corpus_en, corpus_es, trim=trim).alignments
    min_count = 0  # Min count for words to be used in the model, else UNKNOWN
    word_phrase_passes = 3  # 3; Number of word phrase passes
    word_phrase_delta = 3  # 5; min count for word phrase formula
    word_phrase_threshold = 1e-4  # Threshold for word phrase creation
    corpus = Corpus([corpus_en], word_phrase_passes, word_phrase_delta, word_phrase_threshold, 'phrases-%s' % corpus_en)
    corpus_ = Corpus([corpus_es], word_phrase_passes, word_phrase_delta, word_phrase_threshold, 'phrases-%s' % corpus_es)
    vocab = Vocabulary(corpus, min_count)
    vocab_ = Vocabulary(corpus_, min_count)
    tokens = vocab.indices(corpus)
    tokens_ = vocab_.indices(corpus_)
    token2word = {}
    for key, val in zip(vocab.word_map.values(), vocab.word_map.keys()):
        token2word[key] = val
    token2word_ = {}
    for key, val in zip(vocab_.word_map.values(), vocab_.word_map.keys()):
        token2word_[key] = val
    new_corpus_ = []
    indexerrors = 0
    #print(converted_als.__len__())
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


def execute_emssg():
    start = time.time()
    spanish_corpus = "tokenized_es"
    english_corpus = "tokenized_en"
    # prepositions = get_prepositions("prepositions")  OBSOLETE: prepositions now in vocab.prepositions
    aligned_file = "bigfiles/aligned_file"
    output_file = emssg(english_corpus, corpus_es=spanish_corpus, alignment_file=aligned_file, dim=100, epochs=5, enriched=True)
    # Evaluate with specific similarity score: "globalSim", "avgSim", "avgSimC" or "localSim"
    evaluate(output_file, "localSim", enr=True)
    end = time.time()
    print("\nIt took: " + str(round((end-start)/60)) + "min to run.")


def execute_mssg():
    start = time.time()
    dimension = 100
    enrich = False
    if enrich: enr = "enr_"
    else: enr = "not_enr_"
    english_corpus = "tokenized_en"
    # prepositions = get_prepositions("prepositions")  OBSOLETE: prepositions now in vocab.prepositions
    import cProfile
    #cProfile.run('emssg("tokenized_en", epochs=1, dim=100, enriched=False, trim=3000)')
    output_file = emssg(english_corpus, epochs=10, dim=dimension, enriched=enrich, trim=3000)
    # Evaluate with specific similarity score: "globalSim", "avgSim", "avgSimC" or "localSim"
    #evaluate("BEST_" + output_file, "localSim", sense_files=["BEST_" + enr + "SENSES_0", "BEST_" + enr + "SENSES_1"])
    end = time.time()
    print("\nIt took: " + str(round((end-start)/60)) + "min to run.")


def execute_sg():
    start = time.time()
    dimension = 100
    english_corpus = "tokenized_en"
    #output_file = (english_corpus, dimension, epochs=40)
    # output_file = gensim_sg(english_corpus, "gensim_embs")
    # Evaluate with specific similarity score: "globalSim", "avgSim", "avgSimC" or "localSim"
    #evaluate(output_file, "globalSim")
    end = time.time()
    print("\nSkip-gram took: " + str(round((end-start)/60)) + "min to run.")


if __name__ == '__main__':
    """
    Before running: 
    > check trim value
    > check number of epochs(5), dimension(100)
    > check number of senses(2) in emssg()
    > check number of sense filenames in emssg()
    > check alpha(0.01), window(5), min_count(3)
    > if enriched: check trim value in function get_alignments (trim=3000)
    """
    # pD = PreprocessData()
    # pD.preprocess_data()
    # execute_sg()
    # execute_esg()
    # execute_emssg()
    execute_mssg()
