# -*- coding: utf-8 -*-

from __future__ import print_function

import numpy as np
import theano
import theano.tensor as T
import os, glob
import time
import PrepDataReader
import PrepValidation
import SenseExtension
import keras
from keras import layers
from keras.layers import Flatten
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint
from keras.layers import Dense, Input
from keras.layers import Embedding
from keras.models import Model
from keras.initializers import Constant


def run_mlp(config):
    os.chdir("./src/")
    params = config["MLP"]
    senses = params["apply senses"]
    global_embs = params["global embeddings"]
    sense_files = params["sense embeddings"]

    WINDOW_SIZE = 2  # 2 to the left, 2 to the right
    EMBEDDING_DIM = params["dimension"]
    MAX_NUM_WORDS = 1000000

    trainFile = 'STREUSLE/streusle_train.conll'
    devFile = 'STREUSLE/streusle_dev.conll'
    testFile = 'STREUSLE/streusle_test.conll'
    MODEL_DIR = 'models'
    DIR = './'
    print("Preps with Keras with %s" % theano.config.floatX)

    #####################
    # Extract the data
    #####################
    print("Extract data and create matrices")
    train_sentences = PrepDataReader.readFile(trainFile)
    dev_sentences = PrepDataReader.readFile(devFile)
    test_sentences = PrepDataReader.readFile(testFile)

    if senses:
        print("Integrate senses to data")
        SEtrain = SenseExtension.SenseExtension(global_embs, sense_files, train_sentences, dim=EMBEDDING_DIM)
        SEdev = SenseExtension.SenseExtension(global_embs, sense_files, dev_sentences, dim=EMBEDDING_DIM)
        SEtest = SenseExtension.SenseExtension(global_embs, sense_files, test_sentences, dim=EMBEDDING_DIM)

        train_sentences = SEtrain.integrate_senses_to_data()
        dev_sentences = SEtest.integrate_senses_to_data()
        test_sentences = SEdev.integrate_senses_to_data()

        #####################
        # Extract the vocab
        #####################

        print("Extract the vocab from the embeddings file")
        # retrieve embeddings from sense and global files, generate word2Idx
        embeddings, word2Idx = SEtrain.w2emb, SEtrain.w2Idx

    if not senses:
        word2Idx = {}  # Maps a word to the index in the embeddings matrix

        embeddings = {}  # Embeddings matrix
        embeddings["PADDING"] = np.zeros(EMBEDDING_DIM)
        with open(global_embs, 'r') as fIn:
            idx = 0
            for line in fIn:
                values = line.split()
                word = values[0]
                coefs = np.asarray(values[1:], dtype='float32')
                embeddings[word] = coefs
                word2Idx[word] = idx
                idx += 1
        word2Idx["PADDING"] = idx+1
    print('Found %s word vectors.' % len(word2Idx))
    print(word2Idx["UNKNOWN"])
    # Create a mapping for our labels
    label2Idx = {'_': 0}
    idx = 1

    for superSense in ['Affector', 'Attribute', 'Circumstance', 'Configuration', 'Co-Participant', 'Experiencer',
                       'Explanation', 'Manner', 'Place', 'Stimulus', 'Temporal', 'Undergoer']:
        label2Idx[superSense] = idx
        idx += 1

    # Inverse label mapping
    idx2Label = {v: k for k, v in label2Idx.items()}

    # Casing matrix
    caseLookup = {'0': 0, '1': 1}

    # POS matrix (English)
    posLookup = {'.': 0, 'ADD': 6, 'ADJ': 1, 'ADP': 2, 'ADV': 3, 'CONJ': 4, 'DET': 5, 'NOUN': 6, 'NUM': 7, 'PRON': 8,
                 'PRT': 9, 'VERB': 10, 'X': 11, '\'\'': 0, '(': 0, ')': 0, ',': 0, '.': 0, ':': 0, 'CC': 4, 'CD': 7,
                 'DT': 5, 'EX': 5, 'FW': 11, 'HT': 11, 'IN': 2, 'JJ': 1, 'JJR': 1, 'JJS': 1, 'LRB': 0, '-LRB-': 0, 'RRB': 0,
                 '-RRB-': 0, 'LS': 11, 'MD': 10, 'NFP': 11, 'NN': 6, 'NNP': 6, 'NNPS': 6, 'NNS': 6, 'NONE': 11, 'O': 11,
                 'PDT': 5, 'POS': 9, 'PRP': 8, 'PRP$': 8, 'RB': 3, 'RBR': 3, 'RBS': 3, 'RP': 9, 'RT': 0, 'SYM': 11,
                 'TD': 11, 'TO': 9, 'UH': 11, 'URL': 11, 'USR': 6, 'VB': 10, 'VBD': 10, 'VBG': 10, 'VBN': 10, 'VBP': 10,
                 'VBZ': 10, 'VPP': 10, 'WDT': 5, 'WH': 11, 'WP': 8, 'WRB': 3, 'PADDING': 49, 'other': 50, '$': 50, '``': 50,
                 'WP': 50, 'WP$': 50}

    # dependency labels (English)
    depLookup = {'ADV': 1, 'AMOD': 2, 'APPO': 3, 'CONJ': 4, 'COORD': 5, 'DEP': 6, 'DEP-GAP': 7, 'DIR': 8, 'DTV': 9,
                 'EXT': 10, 'EXTR': 11, 'GAP-LOC': 12, 'GAP-OBJ': 13, 'GAP-PRD': 14, 'GAP-SBJ': 15, 'GAP-TMP': 16,
                 'HMOD': 17, 'HYPH': 18, 'IM': 19, 'LGS': 20, 'LOC': 21, 'LOC-PRD': 22, 'MNR': 23, 'NAME': 24, 'NMOD': 25,
                 'OBJ': 26, 'OPRD': 27, 'P': 28, 'PMOD': 29, 'POSTHON': 30, 'PRD': 31, 'PRD-PRP': 32, 'PRN': 33, 'PRP': 34,
                 'PRT': 35, 'PUT': 36, 'ROOT': 37, 'SBJ': 38, 'SUB': 39, 'SUFFIX': 40, 'TITLE': 41, 'TMP': 42, 'VC': 43,
                 'VOC': 44, 'PADDING': 45, 'other': 46}

    posMatrix = np.identity(len(posLookup), dtype='float32')
    depMatrix = np.identity(len(depLookup), dtype='float32')
    caseMatrix = np.identity(len(caseLookup), dtype='float32')


    ###
    print('Preparing embedding matrix.')

    # prepare embedding matrix
    num_words = min(MAX_NUM_WORDS, len(word2Idx) + 1)
    embedding_matrix = np.zeros((num_words, EMBEDDING_DIM))
    for word, i in word2Idx.items():
        if i >= MAX_NUM_WORDS:
            continue
        embedding_vector = embeddings.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector

    ###

    # Create numpy arrays
    train_x, train_pos_x, train_dep_x, train_case_x, train_y = PrepDataReader.createNumpyArrayWithCasing(train_sentences,
                                                                                                         WINDOW_SIZE,
                                                                                                         word2Idx,
                                                                                                         label2Idx,
                                                                                                         posLookup,
                                                                                                         depLookup,
                                                                                                         caseLookup)
    dev_x, dev_pos_x, dev_dep_x, dev_case_x, dev_y = PrepDataReader.createNumpyArrayWithCasing(dev_sentences, WINDOW_SIZE,
                                                                                               word2Idx, label2Idx,
                                                                                               posLookup, depLookup,
                                                                                               caseLookup)
    test_x, test_pos_x, test_dep_x, test_case_x, test_y = PrepDataReader.createNumpyArrayWithCasing(test_sentences,
                                                                                                    WINDOW_SIZE, word2Idx,
                                                                                                    label2Idx, posLookup,
                                                                                                    depLookup, caseLookup)

    #####################################
    #
    # Create the Keras Network
    #
    #####################################


    # Create the train and predict_labels function
    n_hidden = 150
    n_in = 2 * WINDOW_SIZE + 3
    p_in = n_in + 1
    d_in = 3
    c_in = 1
    n_out = len(label2Idx)

    number_of_epochs = 25
    batch_size = 50

    print("units, epochs, batch: ", EMBEDDING_DIM, number_of_epochs, batch_size)

    x = T.imatrix('x')  # the data, one word(prep)+context per row
    y = T.ivector('y')  # the labels are presented as 1D vector of [int] labels

    # load pre-trained word embeddings into an Embedding layer
    # either set trainable = True|False
    embedding_layer = Embedding(input_dim=num_words,
                                output_dim=EMBEDDING_DIM,
                                embeddings_initializer=Constant(embedding_matrix),
                                input_length=n_in,
                                trainable=True)
    print('Training model.')

    words = Input(shape=(n_in,), dtype='int32')
    e_words = embedding_layer(words)
    # e_words = layers.Dropout(0.1)(e_words)
    w_flat = Flatten()(e_words)

    pos = Input(shape=(p_in,), dtype='int32', name='pos')
    # This embedding layer will encode the pos input sequence
    # into a sequence of dense 512-dimensional vectors.
    e_pos = Embedding(output_dim=posMatrix.shape[1], input_dim=len(posLookup), input_length=p_in)(pos)
    # e_pos = layers.Dropout(0.1)(e_pos)
    p_flat = Flatten()(e_pos)

    dep = Input(shape=(d_in,), dtype='int32', name='dep')
    e_dep = Embedding(output_dim=depMatrix.shape[1], input_dim=len(depLookup), input_length=d_in)(dep)
    # e_dep = layers.Dropout(0.1)(e_dep)
    d_flat = Flatten()(e_dep)

    case = Input(shape=(c_in,), dtype='int32', name='case')
    e_case = Embedding(output_dim=caseMatrix.shape[1], input_dim=len(caseLookup), input_length=c_in)(case)
    # e_case = layers.Dropout(0.1)(e_case)
    c_flat = Flatten()(e_case)

    merged = keras.layers.concatenate([w_flat, p_flat, d_flat, c_flat])  # , e_dep, e_case])

    # first layer
    merged = Dense(output_dim=n_hidden, init='glorot_uniform', activation='relu')(merged)
    merged = layers.Dropout(0.3)(merged)

    # softmax layer
    output = Dense(output_dim=n_out, init='glorot_uniform', activation='softmax')(merged)

    model = Model(inputs=[words, pos, dep, case], outputs=[output])

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

    print(model.summary())

    train_y_cat = np_utils.to_categorical(train_y, n_out)
    dev_y_cat = np_utils.to_categorical(dev_y, n_out)
    test_y_cat = np_utils.to_categorical(test_y, n_out)

    start_time = time.time()

    # save best model (=> weights)
    checkpoint = ModelCheckpoint(filepath=os.path.join(MODEL_DIR, "model-{epoch:02d}.h5"), save_best_only=False)

    history = model.fit([train_x, train_pos_x, train_dep_x, train_case_x], train_y_cat, epochs=number_of_epochs,
                        batch_size=batch_size, verbose=2, shuffle=True, callbacks=[checkpoint],
                        validation_data=([dev_x, dev_pos_x, dev_dep_x, dev_case_x], [dev_y_cat]))

    print("%.2f sec for training" % (time.time() - start_time))

    # list all data in history
    print(history.history['acc'])
    print(history.history['val_acc'])

    # save the model architecture
    # model_json = model.to_json()
    # open('architecture.json', 'w').write(model_json)

    # load models
    best_models = glob.glob('models/*')

    for modelfile in best_models:
        model.load_weights(os.path.join(DIR, modelfile))
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
        e, rest = modelfile.split('.')
        e = e.replace('models/model-', '')

        pre_dev, rec_dev, f1_dev = PrepValidation.compute_f1(
            model.predict([dev_x, dev_pos_x, dev_dep_x, dev_case_x], batch_size=batch_size, verbose=0), dev_y, idx2Label, e, "dev")
        pre_test, rec_test, f1_test = PrepValidation.compute_f1(
            model.predict([test_x, test_pos_x, test_dep_x, test_case_x], batch_size=batch_size, verbose=1), test_y,
            idx2Label, e, "test")
        print("%s epoch: prec, rec, F1 on dev: %f %f %f, prec, rec, F1 on test: %f %f %f" % (
        e, pre_dev, rec_dev, f1_dev, pre_test, rec_test, f1_test))
        print()
