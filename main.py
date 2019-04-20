# -*- coding: utf-8 -*-
"""
@author: Yoalli García
"""
import json
from src import emssg, w2v_skipgram, word_sim
from Preprocessing import preprocessData
from src.PrepClassifier import run_mlp

if __name__ == '__main__':
    with open('config.json') as json_data_file:
        config = json.load(json_data_file)

    # To preprocess a Europarl parallel corpus, execute:
    # preprocessData.preprocess_europarl(config)

    # To run the skip-gram algorithm, execute:
    # w2v_skipgram.skip_gram(config)

    # To run the MSSG or EMSSG algorithm, execute:
    # emssg.execute_emssg_or_mssg(config)

    # To calculate Spearman correlations for all similarity scores, execute:
    # word_sim.calculate_spearmans_for_all_similartities(config)

    # To evaluate on the preposition classification system, execute:
    run_mlp(config)

    # To plot the nearest context words or the nearest sense words for a sense word's senses, execute:
    # word_sim.plot_nearest_context_words(config, "commission")
    # word_sim.plot_nearest_sense_words(config, "commission")
