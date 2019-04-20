import time
import math
import os
import numpy as np
from src.word_sim import evaluate


class Corpus:
    def __init__(self, filename):
        file_pointer = open(filename, 'r')

        all_tokens = []

        for line in file_pointer:
            line_tokens = line.split()
            for token in line_tokens:
                token = token.lower()

                if len(token) > 1 and token.isalnum():
                    all_tokens.append(token)

        print("\rCorpus read.")

        file_pointer.close()

        self.tokens = all_tokens

        self.save_to_file(filename)

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
            i += 1

        print("\rPreprocessed input file written")

        filepointer.close()

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

    def build_words(self, corpus):
        words = []
        word_map = {}

        for token in corpus:
            if token not in word_map:
                word_map[token] = len(words)
                words.append(Word(token))
            words[word_map[token]].count += 1

        print("\rVocabulary built: %d" % len(words))

        self.words = words
        self.word_map = word_map # Mapping from each token to its index in vocab

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


def sigmoid(z):
    if z > 6:
        return 1.0
    elif z < -6:
        return 0.0
    else:
        return 1 / (1 + math.exp(-z))


def save(vocab, nn0, filename):
    file_pointer = open(filename, 'w')
    for token, vector in zip(vocab, nn0):
        word = token.word.replace(' ', '_')
        vector_str = ' '.join([str(s) for s in vector])
        file_pointer.write('%s %s\n' % (word, vector_str))
    file_pointer.close()


def log_spearman(spearman_corr, filename):
    with open(filename, "a") as sl:
        sl.write(str(spearman_corr)+"\n")
        sl.close()


def get_context(tokens, token_idx, window):
    current_window = np.random.randint(low=1, high=window + 1)
    context_start = max(token_idx - current_window, 0)
    context_end = min(token_idx + current_window + 1, len(tokens))
    context = tokens[context_start:token_idx] + tokens[token_idx + 1:context_end]
    return context


def skip_gram(config):
    start = time.time()
    os.chdir("./src/")
    input_filename = "tokenized_en"
    params = config["skip-gram"]
    epochs = params["epochs"]
    dim = params["dimension"]
    min_count = params["min_count"] # Min count for words to be used in the model, else UNKNOWN
    learning_rate = params["learning rate"]
    k_negative_sampling = 5  # Number of negative examples
    corpus = Corpus(input_filename)
    # Read train file to init vocab
    vocab = Vocabulary(corpus, min_count)
    table = TableForNegativeSamples(vocab)
    # Max window length
    window = params["window"]  # 5 for large set
    print("Training: %s-%d-%d" % (input_filename, window, dim))
    # Initialize network
    nn0 = np.random.uniform(low=-0.5 / dim, high=0.5 / dim, size=(len(vocab), dim))
    nn1 = np.zeros(shape=(len(vocab), dim))
    # Initial learning rate
    initial_alpha = learning_rate
    # Modified in loop
    alpha = initial_alpha
    tokens = vocab.indices(corpus)
    for epoch in range(epochs):
        print("\rTraining epoch %d..." % epoch)
        for token_idx, token in enumerate(tokens):
            # Randomize window size, where win is the max window size
            context = get_context(tokens, token_idx, window)
            for context_word in context:
                # Init neu1e with zeros
                neu1e = np.zeros(dim)
                classifiers = [(token, 1)] + [(target, 0) for target in table.sample(k_negative_sampling)]
                for target, label in classifiers:
                    z = np.dot(nn0[context_word], nn1[target])
                    p = sigmoid(z)
                    g = alpha * (label - p)
                    neu1e += g * nn1[target]  # Error to backpropagate to nn0
                    nn1[target] += g * nn0[context_word]  # Update nn1
                # Update nn0
                nn0[context_word] += neu1e
        alpha = 0.8 ** epoch * initial_alpha
        save(vocab, nn0, 'SG-Embeddings-%s-%d-%d' % (input_filename, window, dim))
        sp = evaluate('SG-Embeddings-%s-%d-%d' % (input_filename, window, dim), "globalSim")
        log_spearman(sp, "../Spearman_global_skip-gram")
    # Save model to file
    save(vocab, nn0, 'SG-Embeddings-%s-%d-%d' % (input_filename, window, dim))


if __name__ == '__main__':
    start = time.time()
    skip_gram("tokenized_en")
    end = time.time()
    print("\nIt took: " + str(round((end-start)/60)) + "min to run.")

