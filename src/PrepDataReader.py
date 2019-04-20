# -*- coding: utf-8 -*-
"""
createNumpyArrayWithCasing returns the matrices for the word embeddings as well as for the case information
and the Y-vector with the labels
@author: N. Reimers, L. Flekova, Ines
"""
import numpy as np
import re

# from unidecode import unidecode


def readFile(filepath):
    sentences = []
    sentence = []
    prepId = -1

    for line in open(filepath):
        line = line.strip()
        if len(line) == 0:
            if len(sentence) > 0:
                sentences.append(sentence)
                sentence = []
            continue
        splits = line.split('\t')
        # ewtb.r.022533.2:11
        # 1      even    even    RB      _       2       DEP
        # splits1: wordform, splits2: lemma, splits3:pos, splits4: sense, splits5: hd, splits6: deprel
        if (len(splits) > 6):
            sentence.append([splits[1], splits[2], splits[3], splits[5], splits[6], splits[4], int(prepId)])
        # print splits[1], splits[2], splits[3], splits[5], splits[6], splits[4], prepId
        elif len(splits) == 1:
            splits = line.split(' ')
            prepId = splits[1]
            _, prepId = prepId.split(':')
        else:
            print("LENGTH ERROR: ", len(splits), line)
    return sentences


# use lemma for look-up
def getWordIndex(word, lem, unknownIdx, word2Idx):
    unk = 0
    if lem in word2Idx:
        wordIdx = word2Idx[lem]
    elif lem.lower() in word2Idx:
        wordIdx = word2Idx[lem.lower()]
    elif normalizeWord(lem) in word2Idx:
        wordIdx = word2Idx[normalizeWord(lem)]
    else:
        wordIdx = unknownIdx
        unk = 1
    # print "RETURN " + str(unk) + " " + word + " " + str(wordIdx)
    return wordIdx, unk


# use word form, lemma for back-off
def bak_getWordIndex(word, lem, unknownIdx, word2Idx):
    unk = 0
    if word in word2Idx:
        wordIdx = word2Idx[word]
    elif word.lower() in word2Idx:
        wordIdx = word2Idx[word.lower()]
    elif lem in word2Idx:
        wordIdx = word2Idx[lem]
    elif lem.lower() in word2Idx:
        wordIdx = word2Idx[lem.lower()]
    elif normalizeWord(word) in word2Idx:
        wordIdx = word2Idx[normalizeWord(word)]
    elif normalizeWord(lem) in word2Idx:
        wordIdx = word2Idx[normalizeWord(lem)]
    else:
        wordIdx = unknownIdx
        unk = 1
    # print "RETURN " + word + " " + str(wordIdx)
    return wordIdx, unk


# add deprel for prep-head
def addPrepHead(targetWordIdx, sentence, depLookup, depIndices):
    # if depLookup.has_key( sentence[targetWordIdx][4] ):
    if sentence[targetWordIdx][4] in depLookup:
        depIndices.append(depLookup[sentence[targetWordIdx][4]])
    else:
        depIndices.append(depLookup['other'])

    return depIndices


# find id for prep object
def findPrepMod(targetWordIdx, sentence, depLookup):
    # deprel for prep-mod
    modId = -1
    # print targetWordIdx, sentence[targetWordIdx][0], sentence[targetWordIdx][1], sentence[targetWordIdx][4], sentence[targetWordIdx][6]
    for nextId in range(targetWordIdx + 1, len(sentence)):
        # print nextId, sentence[nextId][3]
        if int(sentence[nextId][3]) - 1 == targetWordIdx:
            # 8 others NNS 8 PMOD _ 7
            if modId > -1:
                continue
            if sentence[nextId][4] in depLookup:
                # if (depLookup.has_key( sentence[nextId][4] )):
                modId = nextId
            # print "a) append mod " + sentence[modId][0], sentence[modId][1]

    # noting found? look for postpositions
    if modId == -1:
        for nextId in range(targetWordIdx - 1, 0, -1):
            if int(sentence[nextId][3]) - 1 == targetWordIdx:
                if modId > -1:
                    continue
                if sentence[nextId][4] in depLookup:
                    # if (depLookup.has_key( sentence[nextId][4] )):
                    modId = nextId
                # print "b) append mod " + sentence[modId][0]
    if modId == -1:
        # haven't found anything yet? probably a parser error. take the first N after the prep
        for nextId in range(targetWordIdx + 1, len(sentence)):
            if 'NN' in sentence[nextId][2]:
                if modId > -1:
                    continue
                if sentence[nextId][4] in depLookup:
                    # if (depLookup.has_key( sentence[nextId][4] )):
                    modId = nextId
                # print "c) append mod " + sentence[modId][0]
    if modId == -1:
        # still no PP obj? so let's take the first N to the left
        for nextId in range(targetWordIdx - 1, 0, -1):
            if 'NN' in sentence[nextId][2]:
                if modId > 0:
                    continue
                if sentence[nextId][4] in depLookup:
                    # if (depLookup.has_key( sentence[nextId][4] )):
                    modId = nextId
                # print "d) append mod " + sentence[modId][0]
    # print "RETURN MOD ID " + str(modId)
    return modId


# extract features
# right now we don't really need caseLookup here (we only pass it over in case we will extend it later)
def createNumpyArrayWithCasing(sentences, windowsize, word2Idx, label2Idx, posLookup, depLookup, caseLookup):
    unknownIdx = word2Idx['UNKNOWN']
    paddingIdx = word2Idx['PADDING']

    xMatrix = []
    depMatrix = []
    posMatrix = []
    caseMatrix = []
    yVector = []

    wordCount = 0
    unknownWordCount = 0

    for sentence in sentences:
        targetWordIdx = 0

        for targetWordIdx in range(len(sentence)):
            if targetWordIdx != sentence[targetWordIdx][6]:
                continue
            # Get the context of the target word and map these words to the index in the embeddings matrix
            wordIndices = []
            depIndices = []
            posIndices = []
            caseIndices = []

            # word embeddings for prep, prep-head and prep-mod deprel
            # print targetWordIdx, sentence[targetWordIdx][0], sentence[targetWordIdx][4], depLookup[ sentence[targetWordIdx][4] ]
            # deprel for prep
            if sentence[targetWordIdx][4] in depLookup:
                # if depLookup.has_key( sentence[targetWordIdx][4] ):
                depIndices.append(depLookup[sentence[targetWordIdx][4]])
            else:
                depIndices.append(depLookup['other'])

            # deprel for head of prep
            hdId = int(sentence[targetWordIdx][3]) - 1
            if sentence[hdId][4] in depLookup:
                # if (depLookup.has_key( sentence[hdId][4] )):
                depIndices.append(depLookup[sentence[hdId][4]])
            else:
                depIndices.append(depLookup['other'])

            # pos embedding for head of prep
            if sentence[hdId][2] in posLookup:
                # if (posLookup.has_key( sentence[hdId][2] )):
                posIndices.append(posLookup[sentence[hdId][2]])
            else:
                posIndices.append(posLookup['other'])

            # pos embedding for head of prep-head
            hdHdId = int(sentence[hdId][3]) - 1
            if hdHdId == -1:
                posIndices.append(posLookup['PADDING'])
            elif sentence[hdHdId][2] in posLookup:
                # elif (posLookup.has_key( sentence[hdHdId][2] )):
                posIndices.append(posLookup[sentence[hdHdId][2]])
            else:
                posIndices.append(posLookup['other'])

            # word embedding for head of prep
            prepHeadIdx, unk = getWordIndex(sentence[hdId][1], sentence[hdId][2], unknownIdx, word2Idx)
            wordIndices.append(prepHeadIdx)

            # deprel for prep-mod
            modId = findPrepMod(targetWordIdx, sentence, depLookup)
            if modId == -1:
                depIndices.append(depLookup['PADDING'])
                posIndices.append(posLookup['PADDING'])
                wordIndices.append(unknownIdx)
            else:
                depIndices.append(depLookup[sentence[modId][4]])
                posIndices.append(posLookup[sentence[modId][2]])
                # word embedding for prep mod
                # print "check mod word emb " + sentence[modId][1],sentence[modId][2]
                prepModIdx, unk = getWordIndex(sentence[modId][1], sentence[modId][2], unknownIdx, word2Idx)
                wordIndices.append(prepModIdx)

            # boolean feature: is one of the two words to the right of the prep uppercased?
            isUpper = 0
            for wPosition in range(targetWordIdx + 1, targetWordIdx + windowsize + 1):
                if wPosition >= 0 and wPosition < len(sentence):
                    # print "word position ", wPosition, len(sentence)
                    # print sentence[wPosition][1]
                    nextWrd = sentence[wPosition][0]
                    if nextWrd[0].isupper():
                        isUpper = 1
            # print "append case: ", isUpper
            caseIndices.append(isUpper)

            for wordPosition in range(targetWordIdx - windowsize, targetWordIdx + windowsize + 1):

                if wordPosition < 0 or wordPosition >= len(sentence):
                    wordIndices.append(paddingIdx)
                    posIndices.append(posLookup['PADDING'])
                    continue

                word = sentence[wordPosition][0]
                lem = sentence[wordPosition][1]
                pos = sentence[wordPosition][2]
                dep = sentence[wordPosition][4]
                hd = sentence[wordPosition][3]
                wordCount += 1

                # append (normalised) word to wordIndex
                wordIdx, unk = getWordIndex(word, lem, unknownIdx, word2Idx)

                wordIndices.append(wordIdx)
                unknownWordCount += unk

                if pos in posLookup:
                    # if (posLookup.has_key(pos)):
                    posIndices.append(posLookup[pos])
                else:
                    posIndices.append(posLookup['other'])

            # Get the label and map to int
            # wordIndices are the embedding lookup indices for the 5-words window w-2 w-1 prep w+1 w+2
            labelIdx = label2Idx[sentence[targetWordIdx][5]]  # labelIdx: index of sense label

            # word indices (embedding look-up table) for w-2 w-1 prep w+1 w+2, prep-head, prep-mod
            if len(wordIndices) != 7:
                print("W ERROR ", sentence[targetWordIdx][0], sentence[targetWordIdx][1], wordIndices)
            # pos indices (w-2 - w+2, prep-head, head-head, prep-mod
            if (len(posIndices) != 8):
                print("P ERROR ", sentence[targetWordIdx][0], sentence[targetWordIdx][1], posIndices)
            # dependency label indices (prep, prep-head, prep-mod)
            if (len(depIndices) != 3):
                print("D ERROR ", sentence[targetWordIdx][0], sentence[targetWordIdx][1], depIndices)
            # case info index (w+1, w+2)
            if (len(caseIndices) != 1):
                print("C ERROR ", sentence[targetWordIdx][0], sentence[targetWordIdx][1], caseIndices)

            xMatrix.append(wordIndices)
            depMatrix.append(depIndices)
            # pos label indices (w-2 w-1 prep w+1 w+2)
            posMatrix.append(posIndices)
            # xMatrix.append(np.concatenate((np.concatenate((wordIndices, posIndices)), depIndices)))
            caseMatrix.append(caseIndices)
            yVector.append(labelIdx)

    print("Unknowns: %.2f%%" % (unknownWordCount / (float(wordCount)) * 100))
    return (
    np.asarray(xMatrix), np.asarray(posMatrix), np.asarray(depMatrix), np.asarray(caseMatrix), np.asarray(yVector))


def getCasing(word, caseLookup):
    casing = 'other'

    if word.isdigit():  # Is a digit
        casing = 'numeric'
    if word.islower():  # All lower case
        casing = 'allLower'
    elif word.isupper():  # All upper case
        casing = 'allUpper'
    elif word[0].isupper():  # is a title, initial char upper, then all lower
        casing = 'initialUpper'

    return caseLookup[casing]


def getHead(depLookup):
    return


def multiple_replacer(key_values):
    # replace_dict = dict(key_values)
    replace_dict = key_values
    replacement_function = lambda match: replace_dict[match.group(0)]
    pattern = re.compile("|".join([re.escape(k) for k, v in key_values.items()]), re.M)
    return lambda string: pattern.sub(replacement_function, string)


def multiple_replace(string, key_values):
    return multiple_replacer(key_values)(string)


# from https://github.com/LightTable/Python/issues/24
def ensureUtf(s, encoding='utf8'):
    """Converts input to unicode if necessary.

    If `s` is bytes, it will be decoded using the `encoding` parameters.

    This function is used for preprocessing /source/ and /filename/ arguments
    to the builtin function `compile`.
    """
    # In Python2, str == bytes.
    # In Python3, bytes remains unchanged, but str means unicode
    # while unicode is not defined anymore
    if type(s) == bytes:
        return s.decode(encoding, 'ignore')
    else:
        return s


def normalizeWord(line):
    # line = unicode(line, "utf-8") #Convert to UTF8
    line = ensureUtf(line, encoding='utf8')
    line = line.replace(u"„", u"\"")

    line = line.lower();  # To lower case

    # Replace all special charaters with the ASCII corresponding, but keep Umlaute
    # Requires that the text is in lowercase before
    replacements = dict(((u"ß", "SZ"), (u"ä", "AE"), (u"ü", "UE"), (u"ö", "OE")))
    replacementsInv = dict(zip(replacements.values(), replacements.keys()))
    line = multiple_replace(line, replacements)
    #    line = unidecode(line)
    line = multiple_replace(line, replacementsInv)

    line = line.lower()  # Unidecode might have replace some characters, like € to upper case EUR

    line = re.sub("([0-9][0-9.,]*)", '0', line)  # Replace digits by NUMBER

    return line.strip();
