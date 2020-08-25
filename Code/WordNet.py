from os import remove
from os.path import isfile, dirname, abspath

from nltk import word_tokenize
from nltk.corpus import wordnet
from numpy.random import randint

language_output = dirname(dirname(abspath(__file__))) + "/Code/Allred/wordnet.txt"
language = dirname(dirname(abspath(__file__))) + "/Code/Allred/language.txt"

stop_words = ["a", "an", "the", "am", "is", "are", "was",
              "were", "be", "being", "been", "in", "to", "and",
              "my", "that", "about", "above", "across", "after",
              "against", "along", "among", "around", "at", "before",
              "behind", "below", "beneath", "beside", "between", "beyond",
              "by", "down", "during", "except", "for", "from", "inside",
              "into", "like", "near", "off", "on", "out", "outside", "over",
              "past", "since", "throught", "throughout", "toward", "under",
              "underneath", "until", "up", "upon", "with", "within", "without"
              "I", "you", "he", "she", "it", "they", "we", "can", "could", "would"
              "should", "shall", "will", "at"]


def sim(word1, word2, lch_threshold=2.65, verbose=False):
    from nltk.corpus import wordnet as wn
    results = []
    for net1 in wn.synsets(word1):
        for net2 in wn.synsets(word2):
            try:
                lch = net1.lch_similarity(net2)
            except:
                continue
            if lch >= lch_threshold:
                results.append((net1, net2))
    if not results:
        return False
    if verbose:
        for net1, net2 in results:
            print(net1)
            print(net1.definition)
            print(net2)
            print(net2.definition)
            print('path similarity:')
            print(net1.path_similarity(net2))
            print('lch similarity:')
            print(net1.lch_similarity(net2))
            print('wup similarity:')
            print(net1.wup_similarity(net2))
            print('-' * 79)
    return True


def run(input_file: str, replacement_rate: float, loop_stop: int):
    with open(input_file, 'r', encoding='utf8', errors='ignore') as open_file:
        data = open_file.read()
        words = word_tokenize(data)
    for word in range(len(words)):
        if words[word].islower() and not stop_words.__contains__(words[word]):
            if randint(0, 100) / 100 < replacement_rate:
                syn = []
                try:
                    for synset in wordnet.synsets(words[word]):
                        for lemma in synset.lemmas():
                            syn.append(lemma.name())
                    new_word = syn[randint(0, len(syn) // 4)]
                    while new_word.__contains__("_"):
                        new_word = new_word.replace("_", " ")
                    semantic_similarity = sim(words[word], new_word)
                    stop_condition = 0
                    while not semantic_similarity and stop_condition < loop_stop:
                        semantic_similarity = sim(words[word], new_word)
                        stop_condition += 1
                    if new_word != words[word] and not stop_words.__contains__(words[word]) and semantic_similarity:
                        data = data.replace(words[word], new_word)
                except:
                    "not an accepted word"
    words = words.__str__()

    if isfile(language):
        remove(language)
    with open(language, 'w+') as open_file:
        open_file.write("wordnet")
    if isfile(language_output):
        remove(language_output)
    with open(language_output, 'w+', encoding='utf8') as open_file:
        open_file.write(data)
    print(language_output)
