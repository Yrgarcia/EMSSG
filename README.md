# EMSSG

This is an attempted reimplementation of the EMSSG model proposed by [Ghanimifard et al. (2015)](https://aclweb.org/anthology//R15-1029).

## Getting Started
After cloning the project, the file structure should not be altered, as some functions depend on it.

### Prerequisites

Navigate into `EMSSG` and make sure you have Python 3.5 and Python 2.7 installed. Install all requirements by pasting the following command into your terminal:

  `pip3 install -r requirements_3.5.txt`
  
  As the classification system only runs on Python2, also do
  
  `pip2 install -r requirements_2.7.txt`

There is a small excerpt of the English-Spanish Europarl corpus available for testing, but if you want to use the full [Europarl corpus](http://www.statmt.org/europarl/) for either Spanish, Finnish, German or Polish (es, de, fi, pl), download, unzip and rename both corpora to `input_en` and e.g. `input_es` and place them in the corresponding language pair folder located in `Preprocessing`. For English-Spanish that is `/Preprocessing/es_en/`.

Download and unzip the [TreeTagger package](http://www.cis.uni-muenchen.de/~schmid/tools/TreeTagger/) for your system inside the EMSSG folder. If you placed it somewhere else you need to specify its relative path in `config.json`, otherwise you don't need to change the path.

Download and unzip the [TreeTagger parameter file](http://www.cis.uni-muenchen.de/~schmid/tools/TreeTagger/) for the second language (Spanish, Finnish, German or Polish) and place it into the corresponding folder (for the included test corpus use the Spanish par file and place it in: `/Preprocessing/es_en/`).

Clone [fast_align](https://github.com/clab/fast_align) into `EMSSG` and compile it by following the instructions on their GitHub page.

### Configure Parameters
#### Enter parameters in `config.json`:

**Preprocessing**
* `language`: language tag of the second language of the parallel corpus. Can be `"es"`, `"fi"`, `"de"` or `"pl"`
* `use pos tags`: `true` if POS-tags should be used for aligning the parallel corpora, otherwise `false`
* `TreeTagger location`: relative path to the folder containing the tagger package
* `fast_align location`: relative path to fast_align folder

**Skip-gram**
* `learning rate`: learning rate alpha
* `epochs`: number of iterations through corpus
* `min_count`: words occurring less than `min_count` times will be excluded from training
* `dimension`: dimensionality of the embeddings

**EMSSG**
* `learning rate`: learning rate alpha
* `epochs`: number of iterations through corpus
* `most common`: number of words you want to train senses for
* `senses`: number of senses you want to train 
* `min_count`: words occurring less than `min_count` times will be excluded from training
* `dimension`: dimensionality of the embeddings
* `enriched`: `false` for MSSG, `true` for EMSSG
* `language`: language tag of the second language of the parallel corpus. Can be `"es"`, `"fi"`, `"de"` or `"pl"`
* `use prepositions`: if `true`, model trains senses for prepositions only; if false, model trains senses for N most common words
* `print cluster counts`: if `true`, print out number of context vectors assigned to each cluster for every token after each iteration

**word_sim**
* `global embeddings`: file location of global embeddings
* `sense embeddings`: list of files containing the sense embeddings, "not_enr_SENSES_K" for MSSG and "enr_SENSES_K" for EMSSG, e.g. `["not_enr_SENSES_0","not_enr_SENSES_1"]`

#### Parameters for MLP in MLP/MLP_config.json
**MLP**
* `apply senses`: if `true`, use multiple embeddings for words in the sense embedding files
* `dimension`: dimensionality of the embeddings
* `global embeddings`: file location of global embeddings, e.g. `"../src/MSSG-tokenized_en-5-300-2"`. Output filename varies depending on the input parameters.
* `sense embeddings`: list of files containing the sense embeddings, "not_enr_SENSES_K" for MSSG and "enr_SENSES_K" for EMSSG, e.g. `["../src/not_enr_SENSES_0","../src/not_enr_SENSES_1"]`

## Running a test

Edit `main.py` and uncomment the corresponding function depending on whether you want to 
* preprocess the parallel corpus
* run the MSSG or EMSSG model, depending on the configuration
* run the skip-gram model
* calculate Spearman correlations for all available similarity scores (globalSim, avgSim, avgSimC, localSim, maxSim) for SCWS and WS-353 test data
* plot the nearest sense or global vectors given a word as input

and run `main.py` with **python3.5**.

For evaluating the embeddings on the MLP classification system see below.

### Preprocess Parallel Corpus
You can either use the excerpt of the English-Spanish parallel corpus provided in `Preprocessing/es_en/` for preprocessing or download a [Europarl Parallel Corpus](http://www.statmt.org/europarl/). Currently, Spanish, Finnish, German and Polish are supported for training the EMSSG. The resulting alignments and tokenized files will be automatically saved in the `/src/` directory for the EMSSG model to use.

### Train embeddings with EMSSG or word2vec skip-gram
Per default, the tokenized input files are in the same directory as `emssg.py` and `skip_gram.py` and named for example `tokenized_es`.
You need to determine the language tag for the second language (es, de, fi or pl) in the configuration file. For the monolingual model, the filename automatically is `tokenized_en`.
The output embedding files will be saved in the `/src/` folder. The output files for the Spearman correlation of each epoch are stored directly in `EMSSG`.

### Evaluate embeddings on the preposition classification system (MLP)
If you want to evaluate the embeddings generated by a previously run EMSSG, MSSG or skip-gram model, copy the name of the output file in `/src/` and paste it into the file path for the global and sense embeddings files in `MLP/MLP_config.json`.
Run `MLP/PrepClassifier.py` with **python2.7**. In order to run tests for multi-sense embeddings, you need to set `apply senses` to `true` and specify a list of sense embedding files, usually `["../src/enr_SENSES_0","../src/enr_SENSES_1"]` for MSSG or `["../src/not_enr_SENSES_0","../src/not_enr_SENSES_1"]` for EMSSG. 

## Built With

* [Thijs Scheepers' word2vec skip-gram algorithm](https://github.com/tscheepers/word2vec)

