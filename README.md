# EMSSG

This is an attempted reimplementation of the EMSSG model proposed by [Ghanimifard et al. (2015)](https://aclweb.org/anthology//R15-1029).

## Getting Started
After cloning the project, the file structure should not be altered, as some functions depend on it.

### Prerequisites

Make sure you have Python 3.5 installed. Install all requirements by pasting the following command into your terminal:

  `pip3 install -r requirements.txt`

In order to be able to use the TreeTagger POS-tagger, download the [tagger package](www.cis.uni-muenchen.de/~schmid/tools/TreeTagger) for your system and specify their location in the config.
Download and unzip the [TreeTagger par file](http://www.cis.uni-muenchen.de/~schmid/tools/TreeTagger/) for either Spanish, Finnish, German or Polish and place it in the corresponding folder (for Spanish: `/Preprocessing/es_en/`).

### Configure Parameters
Enter parameters in `config.json`:

**Preprocessing**
* language: determines the second language from the parallel corpus. Can be "es", "fi", "de" or "pl"
* use pos tags: true if POS-tags should be used for aligning the parallel corpora, otherwise false

**Skip-gram**
* `learning rate`: set the learning rate alpha
* `epochs`: set number of iterations
* `min_count`: words occurring less than _min_count_ times will be excluded from training
* `dimension`: dimensionality of the embeddings

**EMSSG**
* `learning rate`: set the learning rate alpha
* `epochs`: set number of iterations
* `most common`: number N of most common words for which K senses are trained
* `senses`: number of senses K per most common word
* `min_count`: words occurring less than min_count times will be excluded from training
* `dimension`: dimensionality of the embeddings
* `enriched`: `false` for MSSG, `true` for EMSSG
* `language`: determines the second language from the parallel corpus. Can be `"es"`, `"fi"`, `"de"` or `"pl"`
* `use prepositions`: if `true`, model trains senses for prepositions only; if false, model trains senses for N most * common words
* `print cluster counts`: if `true`, print out number of context vectors assigned to each cluster for every token after each iteration

**MLP**
* `apply senses`: if `true`, use multiple embeddings for words in the sense embedding files
* `global embeddings`: file location of global embeddings
* `sense embeddings`: list of files containing the sense embeddings, "not_enr_SENSES_K" for MSSG and "enr_SENSES_K" for EMSSG, e.g. `["not_enr_SENSES_0","not_enr_SENSES_1"]`

**word_sim**
* `global embeddings`: file location of global embeddings
* `sense embeddings`: list of files containing the sense embeddings, "not_enr_SENSES_K" for MSSG and "enr_SENSES_K" for EMSSG, e.g. `["not_enr_SENSES_0","not_enr_SENSES_1"]`

## Running a test

Edit `main.py` to determine whether you want to preprocess parallel data, run the (E)MSSG or skip-gram model or evaluate embeddings on the preposition classification system (MLP). You can also calculate Spearman correlations for all available similarity scores (globalSim, avgSim, avgSimC, localSim, maxSim) for SCWS and WS-353 test data by executing `src/word_sim.py`. `word_sim.py` also includes functionalities for plotting the nearest sense and global vectors.

### Preprocess Parallel Corpus
You can either use the excerpt of the English-Spanish parallel corpus provided in `Preprocessing/es_en/` for preprocessing or download a [Europarl Parallel Corpus](http://www.statmt.org/europarl/). Currently, Spanish, Finnish, German and Polish are supported for training the EMSSG. The resulting alignments and tokenized files will be automatically saved in the correct directory for the EMSSG model to use.

### Train embeddings with EMSSG or word2vec skip-gram
The tokenized input files should be located in the same directory as `emssg.py` and `skip_gram.py` and named for example `tokenized_LT`, with _LT_ being the language tag. Language tags for secong languages are determined in the configuration file. For the monolingual model, the filename is `tokenized_en`.

### Evaluate embeddings on MLP
To evaluate the generated embeddings on the preposition classification system, simply specify their file location in `config.json` and run `run_mlp(config)` in `main.py`.If you want to run tests for multi-sense embeddings, you need to set `apply senses` to `true` and specify a list of sense embedding files. 


## Built With

* [Thijs Scheepers' word2vec skip-gram algorithm](https://github.com/tscheepers/word2vec)

