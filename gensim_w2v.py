from gensim.models.word2vec import LineSentence
from gensim.models import Word2Vec
from word_sim import evaluate


def save(vocab, nn0, filename):
    file_pointer = open(filename, 'w')
    for token, vector in zip(vocab, nn0):
        #word = token.word.replace(' ', '_')
        vector_str = ' '.join([str(s) for s in vector])
        file_pointer.write('%s %s\n' % (token, vector_str))
    file_pointer.close()


def gensim_sg(input_file, output_file, N=3000):
    print("Extracting corpus...")
    sentences = LineSentence(input_file, limit=N)
    model = Word2Vec(sentences, size=100, alpha=0.025, window=5, min_count=3, sg=1, iter=25)

    words = list(model.wv.vocab)
    embs = []
    for word in words:
        embs.append(model[word])

    save(words, embs, output_file)
    evaluate(output_file, "globalSim", enr=False)

    return output_file


if __name__ == '__main__':
    gensim_sg('bigfiles/tokenized_en', "GENSIM_embs", N=1000000)
    #pass



