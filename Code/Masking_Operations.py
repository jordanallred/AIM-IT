from os import listdir, urandom
from random import randint
import re
from random import choice, random, shuffle
from nltk import pos_tag
from nltk import word_tokenize
from nltk.corpus import wordnet
from nltk.corpus import wordnet as wn
from nltk.tokenize import sent_tokenize
from numpy.random import randint
from os.path import dirname, abspath
from Code.Dictionary import *
import nltk
from collections import defaultdict
from nltk.tokenize import RegexpTokenizer
from collections import Counter
from mstranslator import Translator
from enum import Enum
from secrets import SystemRandom
import random
import codecs
import json
import string
from os.path import dirname, abspath
import gensim
import nltk
import pattern3.text.en as pattern
from nltk.corpus import wordnet as wn
from nltk.tokenize import MWETokenizer
from nltk.tokenize import RegexpTokenizer

MAX_SEQUENCE_LENGTH = 50
SENTENCE_LENGTH = 19
UNIQUE_W0RDS_RATIO = 0.44
MISSPELLED_WORDS_RATIO = 0.36  # Those are words which are not present in file big.txt
NOUN_RATE = 0.24
VERB_RATE = 0.19
ADJ_RATE = 0.076
ADV_RATE = 0.06
PUNCT_RATE = 0.15
STOPWORDS_RATE = 0.50
WORDS_ALL_CAPITAL_LETTERS_RATIO = 0.02
WORDS_FIRST_CAPITAL_LETTER_RATIO = 0.12

EVALUATION_MEASURES = {
    'average_sentence_length': SENTENCE_LENGTH,
    'unique_words_ratio': UNIQUE_W0RDS_RATIO,
    'misspelled_words_rate': MISSPELLED_WORDS_RATIO,
    'average_noun_rate': NOUN_RATE,
    'average_verb_rate': VERB_RATE,
    'average_adj_rate': ADV_RATE,
    'average_adv_rate': ADJ_RATE,
    'average_punct_rate': PUNCT_RATE,
    'stop_words_ratio': STOPWORDS_RATE,
    'words_all_capital_letters_ratio': WORDS_ALL_CAPITAL_LETTERS_RATIO
}


# Helpers
def sentence_classifier(sentence):
    if not sentence.__contains__(','):
        return 'simple'
    for conjunction in coordinating_conjunctions:
        if sentence.__contains__(', ' + conjunction + ' '):
            return 'compound'
    for adverb in conjunctive_adverbs:
        if sentence.__contains__('; ' + adverb + ','):
            return 'compound'
    if sentence.__contains__(', ') and \
            (any(conjunction in sentence for conjunction in subordinating_conjunctions) or
             any(conjunction.capitalize() in sentence for conjunction in subordinating_conjunctions)):
        return 'complex'
    return 'simple'


def join_simple_sentences(sentence_1, sentence_2):
    if randint(0, 1) is 0:
        return '{0}, {1} {2}'.format(sentence_1[:-1],
                                     coordinating_conjunctions[randint(0, len(coordinating_conjunctions) - 1)],
                                     sentence_2[0].lower() + sentence_2[1:])
    else:
        return '{0}; {1} {2}'.format(sentence_1[:-1],
                                     coordinating_conjunctions[randint(0, len(coordinating_conjunctions) - 1)],
                                     sentence_2[0].lower() + sentence_2[1:])


def split_compound_sentence(sentence):
    for conjunction in coordinating_conjunctions:
        if sentence.__contains__(', ' + conjunction + " "):
            return sentence.split(', ' + conjunction + " ")[0] + ".", \
                   sentence.split(', ' + conjunction + " ")[1].capitalize()


def reverse_complex_sentence(sentence):
    for conjunction in subordinating_conjunctions:
        if sentence.__contains__(conjunction):
            if sentence.index(conjunction) is 0:
                dependent = sentence[:sentence.index(',')]
                independent = sentence[sentence.index(',') + 2:]
                return independent[:-1].capitalize() + ", " + dependent[0].lower() + independent[1:] + "."
            else:
                dependent = sentence[sentence.index(',') + 2:]
                independent = sentence[:sentence.index(',')]
                return dependent[:-1].capitalize() + ", " + independent[0].lower() + independent[1:] + "."


def get_sentence_distribution(sentences):
    num_simple, num_compound, num_complex = 0, 0, 0

    for sentence in sentences:
        if sentence_classifier(sentence) is 'simple':
            num_simple += 1
        elif sentence_classifier(sentence) is 'compound':
            num_compound += 1
        elif sentence_classifier(sentence) is 'complex':
            num_complex += 1

    return {'simple': num_simple / len(sentences),
            'compound': num_compound / len(sentences),
            'complex': num_complex / len(sentences)}


def get_contraction_distribution(text):
    contraction_number, expansion_number = 0, 0
    for contraction in contractions:
        while contraction in text:
            contraction_number += text.count(contraction)
            text = text.replace(contraction, "")
    for expansion in expansions:
        while expansion in text:
            expansion_number += text.count(expansion)
            text = text.replace(expansion, "")
    return {'contraction': contraction_number,
            'expansion': expansion_number}


def sentences_to_text(sentences):
    text = ""
    for sentence in sentences:
        if sentence is None:
            continue
        if punctuations.__contains__(sentence) or stop_punctuations.__contains__(sentence):
            text = text[:-1] + sentence + " "
        elif sentence is '(':
            text += sentence
        else:
            text += sentence + " "
    for punctuation in punctuations:
        if text.__contains__(punctuation + 'i '):
            text = text.replace(punctuation + 'i', punctuation + 'I')
        elif text.__contains__('i' + punctuation):
            text = text.replace('i' + punctuation, 'I' + punctuation)
    while text.__contains__(' i '):
        text = text.replace(' i ', ' I ')
    return text


def get_jaccard_coefficient(text_1, text_2):
    words_1 = word_tokenize(text_1)
    words_2 = word_tokenize(text_2)

    similarities, differences = [], []

    for word in words_1:
        if word in words_2:
            similarities.append(word)
            words_2.remove(word)
        else:
            differences.append(word)

    for word in words_2:
        differences.append(word)

    return len(similarities) / (len(similarities) + len(differences))


def word_similarity(word1, word2, lch_threshold=1.15, verbose=False):
    results = []
    for net1 in wn.synsets(word1):
        for net2 in wn.synsets(word2):
            try:
                lch = net1.lch_similarity(net2)
            except:
                continue
            if lch is None:
                continue
            # The value to compare the LCH to was found empirically.
            # (The value is very application dependent. Experiment!)
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


# Operators
def reverse_sentence_distribution(sentences, distribution):
    simple_distribution = distribution['simple']
    compound_distribution = distribution['compound']

    distribution_change = max(simple_distribution, compound_distribution)

    if distribution_change is simple_distribution:
        for sentence_index in range(len(sentences) - 1):
            if sentence_classifier(sentences[sentence_index]) is 'simple':
                sentence_1 = sentences[sentence_index]
                if sentence_classifier(sentences[sentence_index + 1]) is 'simple':
                    sentence_2 = sentences[sentence_index + 1]
                    sentences[sentence_index] = ""
                    sentences[sentence_index + 1] = join_simple_sentences(sentence_1, sentence_2)
        delete_items = []
        for sentence_index in range(len(sentences)):
            if sentences[sentence_index] is "":  # empty
                delete_items.append(sentence_index)
        for delete_index in range(len(delete_items)):
            del sentences[delete_items[delete_index] - delete_index]
    else:  # distribution_change is compound_distribution
        for sentence_index in range(len(sentences)):
            if sentence_classifier(sentences[sentence_index]) is 'compound':
                sentence_1, sentence_2 = split_compound_sentence(sentences[sentence_index])
                sentences[sentence_index] = sentence_1
                sentences.insert(sentence_index + 1, sentence_2)

    for sentence_index in range(len(sentences)):
        if sentence_classifier(sentences[sentence_index]) is 'complex':
            sentences[sentence_index] = reverse_complex_sentence(sentences[sentence_index])

    return sentences


def reverse_contraction_distribution(text, distribution):
    contraction_distribution = distribution['contraction']
    expansion_distribution = distribution['expansion']

    distribution_change = max(contraction_distribution, expansion_distribution)

    if distribution_change is contraction_distribution:
        for contraction in contractions:
            if contraction in text:
                text = text.replace(contraction, expansions[contractions.index(contraction)])
    else:
        for expansion in expansions:
            if expansion in text:
                text = text.replace(expansion, contractions[expansions.index(expansion)])

    return text


def synonym_substitution(text, jaccard_similarity=0.75, depth=100, n_grams=1, verbose=False):
    original = text
    tokenized_words = word_tokenize(text)
    words = []
    for word in range(len(tokenized_words) - n_grams + 1):
        string_of_words = ""
        for index in range(n_grams):
            if index is 0:
                string_of_words += tokenized_words[word]
            else:
                string_of_words += " " + tokenized_words[word + index]
        words.append(string_of_words)

    shuffle(words)

    for word in range(len(words)):
        semantic_similarity = False
        replacement_word = 0
        test_word = words[word]
        experimental_jaccard = get_jaccard_coefficient(original, text)
        if not test_word.isalpha():
            continue

        if jaccard_similarity > experimental_jaccard:
            break
        if pos_tag(test_word)[0][1] != 'NNP' and pos_tag(test_word)[0][1] != 'NNPS' \
                and test_word not in stop_words and test_word not in contractions and test_word not in expansions:
                syn = []
                for synset in wordnet.synsets(test_word):
                    for lemma in synset.lemmas():
                        syn.append(lemma.name())    # add the synonyms
                if len(syn) == 0:
                    continue
                new_word = syn[replacement_word]
                while not semantic_similarity and new_word.lower() != test_word.lower() and replacement_word < depth - 1 \
                        or pos_tag(test_word)[0][1] != pos_tag(new_word)[0][1]:
                    if replacement_word < len(syn):
                        new_word = syn[replacement_word]
                    else:
                        break
                    while new_word.__contains__("_"):
                        new_word = new_word.replace("_", " ")
                    if new_word.lower() != test_word.lower():
                        semantic_similarity = word_similarity(test_word, new_word)
                    replacement_word += 1
                if new_word.lower() != test_word.lower() and not stop_words.__contains__(test_word) and semantic_similarity \
                        and pos_tag(test_word)[0][1] == pos_tag(new_word)[0][1]:
                    if verbose:
                        print("Original word: " + test_word)
                        print("New word: " + new_word)
                    text = text.replace(test_word, new_word)
    return text


def remove_parenthetical_phrases(text, keep_proper=False):
    if not keep_proper:
        while text.__contains__('(') and text.__contains__(')'):
            left = text.index('(')
            right = text.index(')')
            if left > 0:
                text = text[:left - 1] + text[right + 1:]
            else:
                text = text[:left] + text[right + 1:]
        output_text = text
    else:
        tags = pos_tag(text)
        searching, changes_made = True, False
        left, right = -1, -1
        while searching or changes_made:
            left_found, right_found, changes_made, proper_found = False, False, False, False
            for index in range(len(tags)):
                tag = tags[index]
                if left_found and keep_proper and not right_found:
                    if tag[1] == 'NNP' or tag[1] == 'NNPS':
                        proper_found = True
                if tag[0] is '(' and not left_found:
                    left = index
                    left_found = True
                elif tag[0] is ')' and left_found and not right_found:
                    right = index
                    right_found = True
                if left_found and right_found and not proper_found:
                    tags = tags[:left] + tags[right + 1:]
                    changes_made = True
                    break
                if left_found and right_found and proper_found:
                    left_found, right_found, proper_found = False, False, False
            if not changes_made:
                searching = False
        tokens = []
        for tag in tags:
            tokens.append(tag[0])
        output_text = sentences_to_text(tokens)
    return output_text


def remove_appositive_phrases(text, keep_proper=False):
    tags = pos_tag(text)
    searching, changes_made = True, False
    left, right = -1, -1
    while searching or changes_made:
        left_found, right_found, changes_made, proper_found = False, False, False, False
        for index in range(len(tags)):
            tag = tags[index]
            if left_found and keep_proper and not right_found:
                if tag[1] == 'NNP' or tag[1] == 'NNPS':
                    proper_found = True
            if tag[0] is ',' and not left_found:
                left = index
                left_found = True
            elif tag[0] is ',' and left_found and not right_found:
                right = index
                right_found = True
            if left_found and right_found and not proper_found:
                tags = tags[:left] + tags[right + 1:]
                changes_made = True
                break
            if left_found and right_found and proper_found:
                left_found, right_found, proper_found = False, False, False
        if not changes_made:
            searching = False
    tokens = []
    for tag in tags:
        tokens.append(tag[0])
    output_text = sentences_to_text(tokens)
    return output_text


def change_sentence_distribution(text):
    sentence_list = sent_tokenize(text)
    sentence_distribution = get_sentence_distribution(sentence_list)
    reversed_sentence_distribution = reverse_sentence_distribution(sentence_list, sentence_distribution)
    reversed_sentence_text = sentences_to_text(reversed_sentence_distribution)
    return reversed_sentence_text


def change_contraction_distribution(text):
    contraction_distribution = get_contraction_distribution(text)
    reversed_contraction_text = reverse_contraction_distribution(text, contraction_distribution)
    return reversed_contraction_text


def remove_discourse_markers(text):
    for phrase in discourse_markers:
        if text.__contains__(', ' + phrase + ','):
            text.replace(', ' + phrase + ',', '')
        if text.__contains__(phrase + ','):
            text.replace(phrase + ',', '')
    return text


def add_british_noise(text):
    british, american = False, False
    for american_word in american_words:
        if text.__contains__(american_word):
            american = True
            break
    for british_word in british_words:
        if text.__contains__(british_word):
            british = True
            break
    if american or not british:
        for index in range(len(american_words)):
            american_word = american_words[index]
            if text.__contains__(american_word):
                text = text.replace(american_word, british_words[index])
    else:
        for index in range(len(british_words)):
            british_word = british_words[index]
            if text.__contains__(british_word):
                text = text.replace(british_word, american_words[index])
    return text


def change_case_mihaylova(text):
    words = word_tokenize(text)
    for index in range(len(words)):
        word = words[index]
        if len(word) > 3:
            words[index] = word.lower()

    return sentences_to_text(words)


def split_merge_sentences_mihaylova(text, text_sentence_length, target_sentence_length):
    if text_sentence_length > target_sentence_length:
        sentences = sent_tokenize(text)
        for index in range(len(sentences)):
            sentence = sentences[index]
            pos_tagged = pos_tag(word_tokenize(sentence))
            # print(pos_tagged)
            noun_found, verb_found, split_sentence = False, False, False
            for pos in pos_tagged:
                if pos[1] == 'NN' or pos[1] == 'NNP' or pos[1] == 'NNS' or pos[1] == 'NNPS':
                    noun_found = True
                if pos[1] == 'VB' or pos[1] == 'VBD' or pos[1] == 'VBG' or pos[1] == 'VBN' or pos[1] == 'VBP' \
                        or pos[1] == 'VBZ':
                    verb_found = True
                if noun_found and verb_found and pos[0] == 'and':
                    new_sentences = sentence.split('and')
                    sentence1 = new_sentences[0]
                    sentence2 = new_sentences[1]
                    if sentence1[-1] == ' ':
                        sentence1 = sentence1[:-1]
                    if sentence2[0] == ' ':
                        sentence2 = 'And' + sentence2
                    else:
                        sentence2 = 'And ' + sentence2
                    sentence1 = sentence1 + '.'
                    sentence2 = sentence2[0].capitalize() + sentence2[1:]
                    del sentences[index]
                    sentences.insert(index, sentence2)
                    sentences.insert(index, sentence1)
        return sentences_to_text(sentences)
    elif text_sentence_length < target_sentence_length:
        mihaylova_punctuation = [',', ';']
        mihaylova_conjunctions = ['and', 'as', 'yet']
        sentences = sent_tokenize(text)
        for index in range(len(sentences)):
            sentence = sentences[index]
            if index == 0:
                sentence = sentence[:-1]
                sentence += mihaylova_punctuation[randint(0, 1)]
            elif index < len(sentences) - 1:
                sentence = mihaylova_conjunctions[randint(0, 2)] + " " + sentence[0].lower() + sentence[1:]
                sentence = sentence[:-1]
                sentence += mihaylova_punctuation[randint(0, 1)]
            elif index == len(sentences) - 1:
                sentence = mihaylova_conjunctions[randint(0, 2)] + " " + sentence[0].lower() + sentence[1:]
            del sentences[index]
            sentences.insert(index, sentence)
        return sentences_to_text(sentences)
    else:
        return text


def change_punctuation(text, text_punctuation_distribution, target_punctuation_distribution):
    if text_punctuation_distribution > target_punctuation_distribution:
        text = text.replace(',', '')
        text = text.replace(';', '')
    elif text_punctuation_distribution < target_punctuation_distribution:
        words = word_tokenize(text)
        pos_tagged = pos_tag(words)
        for index in range(len(pos_tagged)):
            pos = pos_tagged[index]
            if pos[1] == 'IN':
                if randint(0, 9) < 6:
                    words.insert(index, ',')
                else:
                    words.insert(index, ';')
    return text


def firstRegexMore(regex1, regex2, aText):
    re1 = re.findall(regex1, aText, re.IGNORECASE)
    re2 = re.findall(regex2, aText, re.IGNORECASE)
    if len(re1) > len(re2):
        return True
    else:
        return False


def regexMoreInText1(regex, aText1, aText2):
    re1 = re.findall(regex, aText1, re.IGNORECASE)
    re2 = re.findall(regex, aText2, re.IGNORECASE)
    if len(re1) > len(re2):
        return True
    else:
        return False


def randReplaceSingle(obfuscation, toFind, toRepl, aProb):
    splitted = obfuscation.split()
    for aCounter, aToken in enumerate(splitted):
        if random() < aProb:
            splitted[aCounter] = re.sub(toFind, toRepl, aToken)
    return " ".join(splitted)


def randReplaceCompare(original, sameAuthor, obfuscation, toFind, toRepl1, toRepl2, aProb):
    splitted = obfuscation.split()
    if regexMoreInText1(toFind, original, sameAuthor):
        for aCounter, aToken in enumerate(splitted):
            if re.findall(toFind, aToken):
                if random() < aProb:
                    splitted[aCounter] = re.sub(toFind, toRepl1, aToken)
    else:
        for aCounter, aToken in enumerate(splitted):
            if re.findall(toFind, aToken):
                if random() < aProb:
                    splitted[aCounter] = re.sub(toFind, toRepl2, aToken)
    return " ".join(splitted)


def kocher1(text):
    obfuscation = text[:]
    obfuscation = re.sub(r" (is|do|does|did|was|were|could|has|have)n't", r" \1 not", obfuscation)
    obfuscation = re.sub(r" can't", r" can not", obfuscation)
    obfuscation = re.sub(r" - ([^.]*?) - ", r" (\1) ", obfuscation)
    obfuscation = re.sub(r" (is|are|was|were) being ", r" \1 ", obfuscation)
    for aValue in veryInverse.keys():
        obfuscation = re.sub(r" " + aValue + " ", r" very " + veryInverse[aValue] + " ", obfuscation)
    obfuscation = re.sub(r" started (\w{4,})ing ", r" \1ed ", obfuscation)
    obfuscation = re.sub(r" in order to ", r" to ", obfuscation)
    obfuscation = re.sub(r" in fact(,?) ", r" actually\1 ", obfuscation)
    obfuscation = re.sub(r"However(,?) ", r"On the contrary\1 ", obfuscation)
    obfuscation = re.sub(r"([\w]{4,}, [\w]{4,},?) and ([\w]{4,}[^,])", r"\1 as well as \2", obfuscation)
    obfuscation = randReplaceSingle(obfuscation, r"\?", choice(["?", "??", "???"]), 0.5)
    obfuscation = randReplaceSingle(obfuscation, r"\!", choice(["!", "!!", "!!!"]), 0.5)
    obfuscation = re.sub(r" the (\w+) of the (\w+( [A-Z][\w']+ ?)?)", r" the \2 \1", obfuscation)
    obfuscation = randReplaceSingle(obfuscation, r"((\w+)(\w)\3(\w+))", r"\2\3\4", 0.05)
    obfuscation = randReplaceSingle(obfuscation, r"((\w+)(\w)\3(\w+))", r"\2\3\3\3\4", 0.05)
    return obfuscation


def kocher2(text):
    obfuscation = text[:]
    obfuscation = re.sub(r" (is|do|does|did|was|were|could|has|have) not ", r" \1n't ", obfuscation)
    obfuscation = re.sub(r" can not ", r" can't ", obfuscation)
    obfuscation = re.sub(r" \(([^.]*?)\) ", r" - \1 - ", obfuscation)
    obfuscation = re.sub(r" (is|are|was|were) (\w+ed) ", r" \1 being \2 ", obfuscation)
    for aKey in very.keys():
        obfuscation = re.sub(r" very " + aKey + " ", r" " + choice(very[aKey]) + " ", obfuscation)
    obfuscation = re.sub(r" actually(,?) ", r" in fact\1 ", obfuscation)
    obfuscation = re.sub(r"On the contrary(,?) ", r"However\1 ", obfuscation)
    obfuscation = re.sub(r" as well as ", r" and ", obfuscation)
    obfuscation = randReplaceSingle(obfuscation, r"((\w+)(\w)\3(\w+))", r"\2\3\4", 0.05)
    obfuscation = randReplaceSingle(obfuscation, r"((\w+)(\w)\3(\w+))", r"\2\3\3\3\4", 0.05)
    return obfuscation


class BritishAmericanNormalization(object):
    britishToAmerican = defaultdict(dict)
    americanToBritish = defaultdict(dict)

    def __init__(self, **kwargs):
        with open(dirname(dirname(abspath(__file__))) + '/Dictionaries/BritishToAmerican.txt', 'r',
                  encoding='utf-8') as f:
            s = f.read()
            temp = s.splitlines()
            for x in temp:
                split = x.split(',')
                self.britishToAmerican[split[0]] = split[1]
                self.americanToBritish[split[1]] = split[0]

    def create_errors(self, text):
        pos_tagged = pos_tag(text)
        result = []
        for n, tagged_word in enumerate(pos_tagged):
            changed = False
            newWord = tagged_word[0].lower()
            if newWord in self.britishToAmerican:
                rnd = random.choice([True, False])
                if rnd:
                    changed = True
                    newWord = self.britishToAmerican[newWord]
                    result.append((newWord, tagged_word[1]))

            if newWord in self.americanToBritish:
                rnd = random.choice([True, False])
                if rnd:
                    changed = True
                    newWord = self.americanToBritish[newWord]
                    result.append((newWord, tagged_word[1]))

            if not changed:
                result.append(tagged_word)

        return pos_tagged_sentence_to_string(result)

class ErrorCreator(object):
    spellingErrorDictionary = defaultdict(dict)

    def __init__(self, **kwargs):
        with open(dirname(dirname(abspath(__file__))) + '/Dictionaries/common-mistakes.txt', 'r',
                  encoding='utf-8') as f:
            s = f.read()
            temp = s.splitlines()
            for x in temp:
                split = x.split('-')
                key = split[0].strip()
                values = list(map(lambda x: x.strip(), split[1].split(',')))
                self.spellingErrorDictionary[key] = values

    def number_of_possible_mistakes(self, text):
        wordList = words(text)
        possibleMistakes = 0
        for w in wordList:
            if w.lower() in self.spellingErrorDictionary:
                possibleMistakes += 1
        return possibleMistakes

    def create_errors(self, text, errorRate):
        wordList = words(text)
        for n, w in enumerate(wordList):
            if w.lower() in self.spellingErrorDictionary:
                changeRatePecents = errorRate * 100
                rnd = random.randint(0, 100) < changeRatePecents
                if rnd:
                    variants = self.spellingErrorDictionary.get(w)
                    wordList[n] = random.choice(variants)
        return ' '.join(wordList)

    def pos_tag_and_create_errors(self, text, changeRate):
        changeRatePercents = changeRate * 100
        result = []
        pos_tagged = pos_tag(text)
        for tagged_word in pos_tagged:
            if tagged_word[0].lower() in self.spellingErrorDictionary:
                rnd = random.randint(0, 100) < changeRatePercents
                if rnd:
                    variants = self.spellingErrorDictionary.get(tagged_word[0])
                    if variants:
                        changed_word = (random.choice(variants), tagged_word[1])
                        result.append(changed_word)
                    else:
                        result.append(tagged_word)
            else:
                result.append(tagged_word)

        return pos_tagged_sentence_to_string(result)

class FillerWords(object):
    filler_words = []

    def __init__(self, **kwargs):
        with open(dirname(dirname(abspath(__file__))) + '/Dictionaries/discourse_markers.txt', 'r',
                  encoding='utf-8') as f:
            s = f.read()
            temp = s.splitlines()
            for x in temp:
                self.filler_words.append(x)

    # If the sentence starts with a filler word, remove it
    # If clear_inside_sentence is given - clear also such words inside the sentence
    # If split_sentence is set - split the sentence in two at the position of this word
    def clear_filler_words(self, text, clear_inside_sentence, split_sentence):
        sentences = sent_tokenize(text)
        for sentence in sentences:
            changed_sentence = sentence
            for fw in self.filler_words:
                if sentence.lower().startswith(fw):
                    changed_sentence = re.sub(fw, '', sentence, flags=re.IGNORECASE)
            if clear_inside_sentence and any(fw in sentence.lower() for fw in self.filler_words):
                for fw in self.filler_words:
                    changed_sentence = re.sub(fw, '.' if split_sentence else '', changed_sentence,
                                              flags=re.IGNORECASE)
            changed_sentence = turn_first_char_uppercase(changed_sentence)

            text = text.replace(sentence, changed_sentence)
        return text

    def insert_random(self, text):
        sentences = sent_tokenize(text)
        for sentence in sentences:
            # changeRatePecents = changeRate * 100
            # rnd = random.randint(0, 100) < changeRatePecents
            rnd = random.choice([False, True])
            if rnd:
                changed_sentence = random.choice(self.filler_words) + ' ' + sentence
                changed_sentence = turn_first_char_uppercase(changed_sentence)
                text = text.replace(sentence, changed_sentence)
        return text

class ParaphraseCorpus(object):
    obfuscationCorpus = defaultdict(dict)

    def __init__(self, **kwargs):
        with open(dirname(dirname(abspath(__file__))) + '/Dictionaries/phrasal-corpus.txt', 'r',
                  encoding='utf-8') as f:
            s = f.read()
            temp = s.splitlines()
            for x in temp:
                split = x.split(' - ')
                self.obfuscationCorpus[split[0]] = split[1]

    def obfuscate(self, text):  # TODO: What should be the change rate ? Now it is effectively 0.5
        if any(w in text for w in self.obfuscationCorpus.keys()):
            for key, value in self.obfuscationCorpus.items():
                if text.find(' ' + key + ' ') > -1:
                    rnd = random.choice([True, False])
                    if rnd:
                        # print('----- Replaced: >' + key + '< with >'+value)
                        text = text.replace(' ' + key + ' ', ' ' + value + ' ')
        return text

class Punctuation(object):
    punctuation = defaultdict(dict)
    prepositions = []

    sys_random = {}

    def __init__(self, **kwargs):
        self.sys_random = SystemRandom(urandom(2))
        self.punctuation = [',', ':', ';']
        with open(dirname(dirname(abspath(__file__))) + '/Dictionaries/prepositions.txt', 'r',
                  encoding='utf-8') as f:
            s = f.read()
            temp = s.splitlines()
            for x in temp:
                self.prepositions.append(x)

    def clear_punctuation(self, text, punct):
        return text.replace(punct, '')

    def clear_all_punctuation(self, text, change_rate):
        change_rate = change_rate * 150

        for p in self.punctuation:
            rnd = random.randint(0, 100) < change_rate
            if rnd:
                text = self.clear_punctuation(text, p)
        return text

    """
    Insert random punctuation - comma or semicolon before a preposition
    """

    def insert_random(self, text, change_rate):
        change_rate = change_rate * 150
        sentences = sent_tokenize(text)
        result = []
        for sentence in sentences:
            pos_tagged = pos_tag(sentence)
            # Insert random comma or semicolon before prepositions
            for index, tagged_word in enumerate(pos_tagged):
                rnd = random.randint(0, 100) < change_rate
                if index > 0 and rnd and tagged_word[0].lower() in self.prepositions:
                    random_punct = self.sys_random.choice([',', ';', ',', ','])
                    result.append((random_punct, '.'))
                result.append(tagged_word)

        return pos_tagged_sentence_to_string(result)

    def insert_redundant_symbols(self, text, change_rate):
        change_rate = change_rate * 150
        exclamationPunct = ['!!', '!', '!!!']
        questionPunct = ['???', '?', '??', '?!?', '!?!']
        rnd = random.randint(0, 100) < change_rate
        if rnd:
            text = text.replace('!', self.sys_random.choice(exclamationPunct))
            text = text.replace('?', self.sys_random.choice(questionPunct))
        return text

class SpellCheck(object):
    NWORDS = defaultdict(lambda: 1)

    def __init__(self, **kwargs):
        with open(dirname(dirname(abspath(__file__))) + '/Dictionaries/big.txt', 'r') as content_file:
            content = content_file.read()
        dictionaryWords = words(content)
        trainingData = train(dictionaryWords)
        self.NWORDS = trainingData

    def edits1(self, word):
        splits = [(word[:i], word[i:]) for i in range(len(word) + 1)]
        deletes = [a + b[1:] for a, b in splits if b]
        transposes = [a + b[1] + b[0] + b[2:] for a, b in splits if len(b) > 1]
        replaces = [a + c + b[1:] for a, b in splits for c in alphabet if b]
        inserts = [a + c + b for a, b in splits for c in alphabet]
        return set(deletes + transposes + replaces + inserts)

    def known_edits2(self, word):
        return set(e2 for e1 in self.edits1(word) for e2 in self.edits1(e1) if e2 in self.NWORDS)

    def known(self, words):
        return set(w for w in words if w in self.NWORDS)

    def correct(self, word):
        candidates = self.known([word]) or self.known(self.edits1(word)) or self.known_edits2(word) or [word]
        return max(candidates, key=self.NWORDS.get)

    def pos_tag_and_correct_text(self, text):
        result = []
        pos_tagged = pos_tag(text)
        for tagged_word in pos_tagged:
            # Do not correct puntuation, numbers and known words
            if tagged_word[1] in ('.', 'NUM') or tagged_word[0].lower() in self.NWORDS or not tagged_word[
                0].isalnum():
                result.append(tagged_word)
            else:
                result.append((self.correct(tagged_word[0]), tagged_word[1]))

        return pos_tagged_sentence_to_string(result)

class SymbolReplacement(object):
    symbols = defaultdict(dict)

    def __init__(self, **kwargs):
        with open(dirname(dirname(abspath(__file__))) + '/Dictionaries/symbols.txt', 'r', encoding='utf-8') as f:
            s = f.read()
            temp = s.splitlines()
            for x in temp:
                split = x.split(' - ')
                self.symbols[split[0]] = split[1]

    def replace_symbols(self, text):
        rnd = random.choice([True, False])
        pos_tagged = pos_tag(text)
        result = []
        for n, tagged_word in enumerate(pos_tagged):
            changed = False
            newWord = tagged_word[0].lower()
            if newWord in self.symbols:
                # rnd = random.choice([True, False])
                # if rnd:
                # print('---------CHANGED SYMBOLS---' + newWord + '----------')
                changed = True
                newWord = self.symbols[newWord]
                result.append((newWord, tagged_word[1]))

            if not changed:
                result.append(tagged_word)

        return pos_tagged_sentence_to_string(result)

class DocumentStats(object):
    def __init__(self, document_text, spellcheck, stopwords):
        self.word_count = word_count(document_text)
        self.sentence_count = sentence_count(document_text)
        self.average_sentence_length = average_sentence_length(document_text)
        self.unique_words_count = unique_words_count(document_text)
        self.unique_words_ratio = unique_words_ratio(document_text)
        self.misspelled_words_rate = misspelled_words_rate(document_text, spellcheck)
        self.pos_rate = pos_ratio(document_text)
        self.average_noun_rate = self.pos_rate['NOUN']
        self.average_verb_rate = self.pos_rate['VERB']
        self.average_adj_rate = self.pos_rate['ADJ']
        self.average_adv_rate = self.pos_rate['ADV']
        self.average_punct_rate = self.pos_rate['PUNCT']
        self.stop_words_count = stopwords.stop_words_sum(document_text)
        self.stop_words_ratio = stopwords.stop_words_ratio(document_text)
        self.words_count = count_words(document_text)
        self.words_first_capital_letter_ratio = capitalized_words_ratio(document_text, False)
        self.words_all_capital_letters_ratio = capitalized_words_ratio(document_text, True)
        self.most_used_words = list(map(lambda x: x[0], self.words_count.most_common(10)))

class TextPart:
    def __init__(self, id, start, end, text):
        self.id = id
        self.start_pos = start
        self.end_pos = end
        self.original_text = text
        self.obfuscated_text = text

def obfuscate_spelling(text, documentStats=None, spellchecker=None, errorCreator=None):
    # If the misspelled words are more than the average - apply spell correction
    # Otherwise - insert common misspellings
    stopwords = StopWords()
    if spellchecker is None:
        spellchecker = SpellCheck()
    if documentStats is None:
        documentStats = DocumentStats(text, spellchecker, stopwords)
    if errorCreator is None:
        errorCreator = ErrorCreator()
    if documentStats.misspelled_words_rate > MISSPELLED_WORDS_RATIO:
        text = spellchecker.pos_tag_and_correct_text(text)
    else:
        errorRate = MISSPELLED_WORDS_RATIO - documentStats.misspelled_words_rate
        text = errorCreator.pos_tag_and_create_errors(text, errorRate)
    return text

def obfuscate_stopwords(text, documentStats=None, stopWords=None):
    spellcheck = SpellCheck()
    if stopWords is None:
        stopWords = StopWords()
    if documentStats is None:
        documentStats = DocumentStats(text, spellcheck, stopWords)
    change_rate = abs(documentStats.stop_words_ratio - STOPWORDS_RATE)
    text = stopWords.obfuscate(text, change_rate)
    return text

def obfuscate_punctuation_count(text, documentStats=None, punctuation=None):
    spellcheck = SpellCheck()
    stopwords = StopWords()

    if documentStats is None:
        documentStats = DocumentStats(text, spellcheck, stopwords)

    if punctuation is None:
        punctuation = Punctuation()

    change_rate = documentStats.average_punct_rate - PUNCT_RATE
    if change_rate > 0:
        text = punctuation.clear_all_punctuation(text, change_rate)
    else:
        change_rate = abs(change_rate)
        text = punctuation.insert_redundant_symbols(text, change_rate)
        text = punctuation.insert_random(text, change_rate)
    return text

'''
Accepts as parameter text which could contain more than one sentence.
Returns text which merges all or some of the sentences.
'''

def merge_sentences(text, merge_all=True):
    sentences = sent_tokenize(text)
    # If there is only one sentence - return it without modifications
    if len(sentences) < 2:
        return text

    result = ''
    prev_edited = False
    for index, sentence in enumerate(sentences):
        # If previous sentence was edited, start current one with lower case
        if prev_edited:
            sentence = turn_first_char_lowercase(sentence)

        # If merge is required for current pair - replace the ending char
        # with randomly selected separators
        if (merge_all or random.choice([True, False])) and index < len(sentences) - 1:
            random_punct = random.choice([',', ';'])
            random_conj = random.choice([' and', ' and', ' and', ' and', '', '', '', ' as', ' yet'])
            sentence = replace_last_char(sentence, random_punct + random_conj)
            prev_edited = True
        else:
            prev_edited = False

        if len(result) > 0:
            result = result + ' '
        result = result + sentence

    result = re.sub(r'\n+', r'\n', result)
    return result

'''
Accepts as parameter list with pos tagged words of a sentence.
Returns as text the sentence split in several sentences.
'''

def split_sentence(pos_tagged_sentence):
    result_pos_tagged = []

    has_noun = False
    has_verb = False
    capitalize_next = False

    for tagged_word in pos_tagged_sentence:
        word = tagged_word[0]
        pos = tagged_word[1]

        if capitalize_next:
            tagged_word = (turn_first_char_uppercase(word), pos)

        if pos == 'NOUN': has_noun = True
        if pos == 'VERB': has_verb = True
        if pos == 'CONJ' and word.lower() == 'and' and has_noun and has_verb:
            # There was a noun and a verb before 'and',
            # split the sentence here
            result_pos_tagged.append(('.', '.'))
            has_noun = False
            has_verb = False
            capitalize_next = True
        else:
            result_pos_tagged.append(tagged_word)
            capitalize_next = False

    return pos_tagged_sentence_to_string(result_pos_tagged)

'''
POS Tags and splits the text.
'''

def pos_tag_and_split_sentences(text):
    pos_tagged = pos_tag(text)
    return split_sentence(pos_tagged)

def obfuscate_sentence_length(text, documentStats=None, fillerWords=None):
    THRESHHOLD = 3  # If difference is no more then 3 words we don't change the sentence length
    spellcheck = SpellCheck()
    stopwords = StopWords()

    if documentStats is None:
        documentStats = DocumentStats(text, spellcheck, stopwords)

    if fillerWords is None:
        fillerWords = FillerWords()

    diff = documentStats.average_sentence_length - SENTENCE_LENGTH
    if abs(diff) <= THRESHHOLD:
        return text
    if diff > 0:
        text = pos_tag_and_split_sentences(text)
        text = fillerWords.clear_filler_words(text, True, True)
    else:
        text = merge_sentences(text)
    return text

def obfuscate_all_uppercase_words(text):
    pos_tagged = pos_tag(text)
    result = []
    for n, tagged_word in enumerate(pos_tagged):
        # Capitalize only the words longer than 3 symbols, the rest could be abbreviations
        if tagged_word[0].isupper() and len(tagged_word[0]) > 3:
            result.append((tagged_word[0].capitalize(), tagged_word[1]))
        else:
            result.append(tagged_word)
    return pos_tagged_sentence_to_string(result)

'''
Removes the adjectives and adverbs from sentence.
Parameters:
pos_tagged_sentence - the list with pos-tagged words of the sentence to be processed
leave_one - if True  - one adjective or adverb will be left if there is a sequence of more
          - if False - all adjectives or adverbs are removed before NOUN or VERB
                     - when other POS - one is left (otherwise meaning is lost)

'''

def remove_all_adjectives(text, changeRate, remove_pos_tag='ADJ', next_pos_tag='NOUN', leave_one=False):
    '''
    Remove all adjectives in the text.
    Find a sequence of adjectives and remove them and their connecting words.
    If the adjectives sequence was the end of the sentence, leave one of them.
    '''
    changeRatePercents = changeRate * 100
    rnd = random.randint(0, 100) < changeRatePercents
    if not rnd:
        # Do not transform the text if the adjectives should not be removed
        return text

    result = []
    first_adjective_from_sequence = ()
    non_adj_word_from_sequence = ()

    pos_tagged_sentence = pos_tag(text)

    for tagged_word in pos_tagged_sentence:
        word = tagged_word[0]
        pos_type = tagged_word[1]
        if pos_type in [remove_pos_tag] and word != 'not' and word != "n't" and not (
        word.startswith('wh')):  # and len(first_adjective_from_sequence) <= 0:
            first_adjective_from_sequence = tagged_word
        elif pos_type == 'CONJ' and word == 'and':
            non_adj_word_from_sequence = tagged_word
        elif pos_type in [next_pos_tag]:
            if leave_one and len(first_adjective_from_sequence) > 0:
                result.append(first_adjective_from_sequence)
                if len(non_adj_word_from_sequence) > 0:
                    result.append(non_adj_word_from_sequence)
            result.append(tagged_word)
            first_adjective_from_sequence = ()
            non_adj_word_from_sequence = ()
        elif pos_type not in [remove_pos_tag, next_pos_tag, 'CONJ'] and word != ',':
            if len(first_adjective_from_sequence) > 0:
                # If there is an adjective on the pipe, add it as it would lose the meaning
                # The house was old and the tree was green.
                result.append(first_adjective_from_sequence)
                if len(non_adj_word_from_sequence) > 0:
                    result.append(non_adj_word_from_sequence)
            result.append(tagged_word)
            first_adjective_from_sequence = ()
            non_adj_word_from_sequence = ()
        elif pos_type == '.':
            if word in ('.', '?', '!'):
                if len(first_adjective_from_sequence) > 0:
                    # If there is an adjective on the pipe, add it as it would lose the meaning
                    # The house was old and the tree was green.
                    result.append(first_adjective_from_sequence)
                    # if len(non_adj_word_from_sequence) > 0:
                    #    result.append(non_adj_word_from_sequence)
                    first_adjective_from_sequence = ()
                    non_adj_word_from_sequence = ()
            result.append(tagged_word)
        else:
            result.append(tagged_word)

    return pos_tagged_sentence_to_string(result)

def obfuscate_pos_count(text, document_stats):
    # We only have removal of adjectives
    if document_stats.average_adj_rate > ADJ_RATE:
        changeRate = document_stats.average_adj_rate - ADJ_RATE
        text = remove_all_adjectives(text, changeRate, 'ADJ', 'NOUN',
                                                                random.choice([True, False]))
    if document_stats.average_adv_rate > ADV_RATE:
        changeRate = document_stats.average_adv_rate - ADV_RATE
        text = remove_all_adjectives(text, changeRate, 'ADV', 'VERB', True)
    return text

def obfuscate_uppercase(text):
    text = obfuscate_all_uppercase_words(text)
    return text

def replace_most_used_words(text, wordCount, changeRate, mostUsedWords=None):
    result = []
    # As very little words were replaced in the experiments, increase the change rate
    changeRatePercents = changeRate * 100  # * 1.5

    pos_tagged = pos_tag(text)
    for tagged_word in pos_tagged:
        word = tagged_word[0]
        pt = tagged_word[1]
        rnd = random.randint(0, 100) < changeRatePercents
        rnd50 = random.choice([True, False])

        # Replace the most common words with probability of 50% instead of the given change rate
        replace = (rnd50 and mostUsedWords and word in mostUsedWords) or rnd

        if replace and pt in ('VERB', 'NOUN', 'ADJ', 'ADV') and wordCount[word] > 3:
            # print('Replacing common word: ' + word + ' rate: ' + str(changeRate))
            new_word = get_most_plausible_synonim(word, pt)
            if new_word == word: new_word = get_hypernim(word, pt)
            result.append((new_word, pt))
        else:
            result.append(tagged_word)
    return pos_tagged_sentence_to_string(result)

# Replace rare words with definitions
def replace_rare_words(text, wordCount, changeRate):
    result = []
    changeRatePercents = changeRate * 100
    pos_tagged = pos_tag(text)
    for tagged_word in pos_tagged:
        word = tagged_word[0]
        pt = tagged_word[1]
        rnd = random.randint(0, 100) < changeRatePercents
        if rnd and pt in ('VERB', 'NOUN', 'ADJ', 'ADV') and wordCount[word] == 1:
            # print('Replacing rare word: ' + word + ' rate: ' + str(changeRate))
            new_word = get_definition(word, pt)
            result.append((new_word, pt))
        else:
            result.append(tagged_word)
    return pos_tagged_sentence_to_string(result)

# be cautions may return different form of the word
def get_most_plausible_synonim(word, posTag):
    if posTag not in ['ADJ', 'ADJ_SAT', 'ADV', 'NOUN', 'VERB']:
        return word
    if posTag == 'NOUN':
        posTag = 'n'
    if posTag == 'VERB':
        posTag = 'v'

    t = wn.synsets(word);
    syns = list(filter(lambda x: str(x._pos) == posTag, t))
    if syns and len(syns) > 0:
        difSyns = list(
            filter(lambda x: str(x.lemma_names()[0]) != word and str(x.lemma_names()[0]).find('_') == -1, syns))
        if difSyns and difSyns[0].lemma_names():
            return difSyns[0].lemma_names()[0]
    return word

def get_most_plausible_synonim(word, posTag):
    difSyns = get_synsets_by_pos(word, posTag)
    if difSyns and difSyns[0].lemma_names():
        return difSyns[0].lemma_names()[0]
    return word

def get_definition(word, posTag):
    difSyns = get_synsets_by_pos(word, posTag)
    if difSyns and difSyns[0].definition():
        return difSyns[0].definition()
    return word

def get_synsets_by_pos(word, posTag):
    if posTag not in ['ADJ', 'ADJ_SAT', 'ADV', 'NOUN', 'VERB']:
        return None
    if posTag == 'NOUN':
        posTag = 'n'
    if posTag == 'VERB':
        posTag = 'v'
    if posTag == 'ADJ_SAT':
        posTag = 's'
    if posTag == 'ADV':
        posTag = 'r'
    if posTag == 'ADJ':
        posTag = 'a'
    # todo temp solution

    result = list()
    t = wn.synsets(word)
    syns = [el for el in t if str(el._pos) == posTag]
    if syns and len(syns) > 0:
        filtr = [fl for fl in syns if str(fl.lemma_names()[0]) != word and str(fl.lemma_names()[0]).find('_') == -1]
        result.extend(filtr)
        return result
    return None  # If not synsets are found

def get_hypernim(word, posTag='NOUN'):
    wordSynset = get_synsets_by_pos(word, posTag)
    if wordSynset:
        hypernim = wordSynset[0].hypernyms()
        if hypernim and hypernim[0] and hypernim[0].lemma_names():
            return hypernim[0].lemma_names()[0]
    return word

def obfuscate_unique_words_count(text, documentStats=None):
    '''
    If the unique word count is above average:
    - replace part of the most used nouns, verbs and adjectives with synonims or hypernims

    If the unique word count is below average:
    - replace with explanation one of the nouns, verbs and adjectives used only once
    '''
    spellcheck = SpellCheck()
    stopwords = StopWords()
    if documentStats is None:
        documentStats = DocumentStats(text, spellcheck, stopwords)
    changeRate = documentStats.unique_words_ratio - UNIQUE_W0RDS_RATIO
    if documentStats.unique_words_ratio > UNIQUE_W0RDS_RATIO:
        text = replace_most_used_words(text, documentStats.words_count, changeRate,
                                                         documentStats.most_used_words)
    else:
        text = replace_rare_words(text, documentStats.words_count, -changeRate)
    return text

class Translation:
    """Translation"""

    def translate(self, textToTranslate, languageList=None):
        languages = [Language.Bulgarian, Language.Croatian, Language.Czech, Language.Danish, Language.Dutch,
                     Language.Estonian, Language.Finnish, Language.German, Language.Greek, Language.Hindi,
                     Language.Hungarian, Language.Indonesian, Language.Italian, Language.Japanese, Language.Klingon,
                     Language.Korean, Language.Latvian, Language.Norwegian, Language.Polish, Language.Russian,
                     Language.Spanish, Language.Swahili, Language.Swedish, Language.Turkish, Language.Ukrainian]
        if languageList is None:
            languageList = [languages[random.randint(0, len(languages))]]
        if type(textToTranslate) is not str:
            raise ValueError("The argument textToTranslate must be string")
        # to save our translation limit the api calls are commented. uncoment to enable translation
        translator = Translator('GeorgiKaradjov', 'Y2c414NBMQlVgVPZK7vmFT7WZ/DJ4sKRYsTxG9NAXlQ=')
        tempTranslation = textToTranslate
        languageFrom = Language.English
        for language in languageList:
            if issubclass(type(language), Language):
                tempTranslation = translator.translate(tempTranslation, lang_from=languageFrom.value,
                                                       lang_to=language.value)
                languageFrom = language
            else:
                raise ValueError("You must pass only valid values from Languages enum")

        tempTranslation = translator.translate(tempTranslation, lang_from=languageFrom.value,
                                               lang_to=Language.English.value)
        return tempTranslation

class Language(Enum):
    English = 'en'
    Bulgarian = 'bg'
    Croatian = 'hr'
    Czech = 'cs'
    Danish = 'da'
    Dutch = 'nl'
    Estonian = 'et'
    Finnish = 'fi'
    German = 'de'
    Greek = 'el'
    Russian = 'ru'
    Hindi = 'hi'
    Hungarian = 'hu'
    Indonesian = 'id'
    Italian = 'it'
    Japanese = 'ja'
    Swahili = 'sw'
    Klingon = 'twh'  # live long and prosper
    Korean = 'ko'
    Latvian = 'lv'
    Norwegian = 'no'
    Polish = 'pl'
    Spanish = 'es'
    Swedish = 'sv'
    Turkish = 'tr'
    Ukrainian = 'uk'


def apply(text):
    if re.match(r"(\w+) of (\w+)", text):
        rnd = random.choice([False, True, False])
        if rnd:
            return re.sub(r"(\w+) of (\w+)", r"\2's \1", text)
    return text

def detect_equation(text):
    comparingSymbols = re.compile('.[<>=]+.')
    innerSymbols = re.compile('.[\+\-\*\/]+.')
    comp1 = comparingSymbols.findall(text)
    comp2 = innerSymbols.findall(text)
    if comp1 and comp2:
        return True
    return False

def all_symbols_tokenizer(text):
    tokenizer = RegexpTokenizer(r'\S+')
    return tokenizer.tokenize(text)

def transform_equation(text):
    words = all_symbols_tokenizer(text)
    for n, w in enumerate(words):
        for sym in symbols:
            if sym in w:
                words[n] = words[n].replace(sym, symbols[sym])
    return ' '.join(words)

symbols = {
    '+': ' plus ',
    '-': ' minus ',
    '*': ' multiplied by ',
    '/': ' divided by ',
    '=': ' equals ',
    '>': ' greather than ',
    '<': ' less than ',
    '<=': ' less than or equal to ',
    '>=': ' greater than or equal to ',
}

def replace_numbers(text):
    pos_tagged = pos_tag(text)
    result = []
    for tagged_word in pos_tagged:
        if tagged_word[1] == 'NUM':
            num_to_string = nums_to_words(tagged_word[0])
            result.append((num_to_string, 'NUM'))
        else:
            result.append(tagged_word)
    return pos_tagged_sentence_to_string(result)

'''
Given a CD from POS Tagging,
split it into number strings and then convert them to words.

If a number contains more than 8 numbers, then spell it digit by digit.
(Ex.: it might be a phone number.)
'''

def nums_to_words(num_string):
    result = num_string
    nums = re.split('\D', num_string)
    for num in nums:
        if len(num) > 0:
            word = ''
            if len(num) >= 8:
                for d in num:
                    if len(word) > 0:
                        word = word + '-'
                    word = word + num_to_words(int(d))
            else:
                word = num_to_words(int(num))
            result = result.replace(num, word)
    return result

'''
words = {} convert an integer number into words

Source: http://stackoverflow.com/questions/8982163/how-do-i-tell-python-to-convert-integers-into-words
'''

def num_to_words(num, join=True):
    # if not any(i.isdigit() for i in num):
    #     print('!!!! no digits ' + num)
    #     return num
    # print ('continue ' + num)

    # num = int(num)

    units = ['', 'One', 'Two', 'Three', 'Four', 'Five', 'Six', 'Seven', 'Eight', 'Nine']
    teens = ['', 'Eleven', 'Twelve', 'Thirteen', 'Fourteen', 'Fifteen', 'Sixteen', \
             'Seventeen', 'Eighteen', 'Nineteen']
    tens = ['', 'Ten', 'Twenty', 'Thirty', 'Forty', 'Fifty', 'Sixty', 'Seventy', \
            'Eighty', 'Ninety']
    thousands = ['', 'thousand', 'million', 'billion', 'trillion', 'quadrillion', \
                 'quintillion', 'sextillion', 'septillion', 'octillion', \
                 'nonillion', 'decillion', 'undecillion', 'duodecillion', \
                 'tredecillion', 'quattuordecillion', 'sexdecillion', \
                 'septendecillion', 'octodecillion', 'novemdecillion', \
                 'vigintillion']
    words = []
    if num == 0:
        words.append('Zero')
    else:
        numStr = '%d' % num
        numStrLen = len(numStr)
        groups = int((numStrLen + 2) / 3)
        numStr = numStr.zfill(groups * 3)
        for i in range(0, groups * 3, 3):
            h, t, u = int(numStr[i]), int(numStr[i + 1]), int(numStr[i + 2])
            g = int(groups - (i / 3 + 1))
            if h >= 1:
                words.append(units[h])
                words.append('hundred')
            if t > 1:
                words.append(tens[t])
                if u >= 1: words.append(units[u])
            elif t == 1:
                if u >= 1:
                    words.append(teens[u])
                else:
                    words.append(tens[t])
            else:
                if u >= 1: words.append(units[u])
            if (g >= 1) and ((h + t + u) > 0): words.append(thousands[g] + ',')
    if join: return ' '.join(words)
    return words

def replace_short_forms(text):
    text = replace_short_negation(text)
    text = replace_short_have(text)
    text = replace_short_would_had(text)
    text = replace_short_to_be(text)
    text = replace_short_will(text)
    return text

def replace_short_negation(text):
    text = re.sub("ain't", 'is not', text, flags=re.IGNORECASE)
    return re.sub(r" (.+)n't", r' \1 not', text, flags=re.IGNORECASE)

def replace_short_have(text):
    return re.sub(r" (.+)'ve", r' \1 have', text, flags=re.IGNORECASE)

def replace_short_will(text):
    return re.sub(r" (.+)'ll", r' \1 will', text, flags=re.IGNORECASE)

def replace_short_would_had(text):
    # Note: short 'd could be would or had
    random_replacement = random.choice(['would', 'had'])
    return re.sub(r" (.+)'d", r' \1 ' + random_replacement, text, flags=re.IGNORECASE)

def replace_short_to_be(text):
    text = re.sub("I'm", 'I am', text, flags=re.IGNORECASE)
    text = re.sub(r" (.+)'re", r' \1 are', text, flags=re.IGNORECASE)
    # 's cannot be replaced safe, as it could be expressing possesion
    return text


class StopWords(object):
    stops = []
    stopObfuscation = defaultdict(dict)

    def __init__(self, **kwargs):
        self.stops = set(stopwords.words('english'))
        with open(dirname(dirname(abspath(__file__))) + '/Dictionaries/stopwords-obfuscation.txt', 'r',
                  encoding='utf-8') as f:
            s = f.read()
            temp = s.splitlines()
            for x in temp:
                split = x.split('-')
                self.stopObfuscation[split[0].strip()] = split[1].strip()

    def stop_words_sum(self, text):
        wds = words(text)
        count = 0
        for x in wds:
            if x.lower() in self.stops:
                count += 1
        return count

    def stop_words_ratio(self, text):
        return self.stop_words_sum(text) / word_count(text)

    def count_stopwords(self, text):
        wds = words(text)

        stopwordsfrequency = []
        for x in wds:
            if x.lower() in self.stops:
                stopwordsfrequency.append(x.lower())
        word_dict = Counter(stopwordsfrequency)
        return word_dict

    def obfuscate(self, text, change_rate):
        change_rate_percents = change_rate * 150
        pos_tagged = pos_tag(text)
        result = []
        for n, tagged_word in enumerate(pos_tagged):
            new_word = tagged_word[0].lower()
            replaced = False
            if new_word in self.stopObfuscation:
                rnd = random.randint(0, 100) < change_rate_percents
                if rnd:
                    replaced = True
                    new_word = self.stopObfuscation[new_word]
                    if new_word:
                        result.append((new_word, tagged_word[1]))
            if not replaced:
                result.append(tagged_word)
        return pos_tagged_sentence_to_string(result)

def word_count(text):
    if text:
        tokens = words(text)
        return len(tokens)
    else:
        return 0

def average_sentence_length(text):
    if text:
        sentences = sent_tokenize(text)
        return word_count(text) / len(sentences)
    else:
        return 0

def sentence_count(text):
    if text:
        sentences = sent_tokenize(text)
        return len(sentences)
    else:
        return 0

def unique_words_count(text):
    if text:
        tokens = words(text)
        unique_words = set(tokens)
        return len(unique_words)
    else:
        return 0

def unique_words_ratio(text):
    return unique_words_count(text) / word_count(text)

def misspelled_words_rate(text, spellcheck):
    if text and spellcheck:
        tokens = words(text)
        return len(spellcheck.known(tokens)) / len(tokens)
    else:
        return 0

def count_words(text):
    wds = words(text)
    wordsfrequency = []
    for x in wds:
        wordsfrequency.append(x.lower())
    word_dict = Counter(wordsfrequency)
    return word_dict

def capitalized_words_ratio(text, allLettersCapital=True):
    wds = words(text)
    cap_words_count = 0
    for word in wds:
        if allLettersCapital and word.isupper() or (not allLettersCapital) and word[0].isupper():
            cap_words_count = cap_words_count + 1
    return cap_words_count / len(wds)

def pos_ratio(text):
    if text:
        wc = word_count(text)
        pos_tagged = pos_tag(text)
        noun_count = 0
        verb_count = 0
        adj_count = 0
        adv_count = 0
        punctuation_count = 0

        for tagged_word in pos_tagged:
            word = tagged_word[0]
            pos = tagged_word[1]
            if pos == 'NOUN':
                noun_count = noun_count + 1
            elif pos == 'VERB':
                verb_count = verb_count + 1
            elif pos == 'ADJ':
                adj_count = adj_count + 1
            elif pos == 'ADV':
                adv_count = adv_count + 1
            elif pos == '.':
                punctuation_count = punctuation_count + 1

        return {
            'NOUN': noun_count / wc,
            'VERB': verb_count / wc,
            'ADJ': adj_count / wc,
            'ADV': adv_count / wc,
            'PUNCT': punctuation_count / wc,
        }
    else:
        return 0


def pos_tag(text):
    tokens = nltk.word_tokenize(text)
    pos = nltk.pos_tag(tokens, tagset='universal')
    return pos

def pos_tagged_sentence_to_string(tagged_sentence_list):
    '''
    Input: [('The', 'DET'), ('house', 'NOUN'), ('is', 'VERB'), ('very', 'ADV'), ('big', 'ADJ'), ('.', '.')]
    Output: 'The house is very big.'
    '''
    return ''.join(
        map(lambda x: x[0] if x[1] == '.' and x[0] not in ("\'", '(', '[') else ' ' + x[0], tagged_sentence_list))[
           1:]

def words(text):
    tokenizer = RegexpTokenizer(r'\w+')
    return tokenizer.tokenize(text)

def turn_first_char_lowercase(text):
    text = text.strip()
    if text:
        return text.lower()[:1] + text[1:]
    else:
        return ''

def turn_first_char_uppercase(text):
    text = text.strip()
    if text:
        return text.upper()[:1] + text[1:]
    else:
        return ''

def replace_last_char(text, replacement):
    if text:
        if text.endswith(('.', '?', '!')):
            return text[0:(len(text) - 1)] + replacement
        else:
            return text
    else:
        return ''

def words(text):
    return re.findall('[a-z]+', text.lower())

def train(features):
    model = defaultdict(lambda: 1)
    for f in features:
        model[f] += 1
    return model

alphabet = 'abcdefghijklmnopqrstuvwxyz'


def apply(text):
    if re.match(r"(\w+) of (\w+)", text):
        rnd = random.choice([False, True, False])
        if rnd:
            return re.sub(r"(\w+) of (\w+)", r"\2's \1", text)
    return text

def transform_equation(text):
    words = all_symbols_tokenizer(text)
    for n, w in enumerate(words):
        for sym in symbols:
            if sym in w:
                words[n] = words[n].replace(sym, symbols[sym])
    return ' '.join(words)

def replace_numbers(text):
    pos_tagged = pos_tag(text)
    result = []
    for tagged_word in pos_tagged:
        if tagged_word[1] == 'NUM':
            num_to_string = nums_to_words(tagged_word[0])
            result.append((num_to_string, 'NUM'))
        else:
            result.append(tagged_word)
    return pos_tagged_sentence_to_string(result)

def replace_short_forms(text):
    text = replace_short_negation(text)
    text = replace_short_have(text)
    text = replace_short_would_had(text)
    text = replace_short_to_be(text)
    text = replace_short_will(text)
    return text


# Utilities
def get_metrics(text):
    words = word_tokenize(text)
    vocabulary = []
    word_tokens, punctuation_tokens, stop_word_tokens, uppercase_tokens = 0, 0, 0, 0
    for word in words:
        if word in punctuations or word in stop_punctuations:
            punctuation_tokens += 1
        elif word in stop_words:
            stop_word_tokens += 1
        else:
            in_vocab = False
            if word.isupper():
                uppercase_tokens += 1
            for index in range(len(vocabulary)):
                vocab_word = vocabulary[index]
                if word == vocab_word['word']:
                    vocabulary[index]['count'] += 1
                    in_vocab = True
                    break
            if not in_vocab:
                vocabulary.append(
                    {
                        'word': word,
                        'count': 1
                    }
                )
            word_tokens += 1

    sentence_length_average, sentence_num = 0, 0
    for sentence in sent_tokenize(text):
        sentence_length = 0
        for word in word_tokenize(sentence):
            if word not in punctuations and word not in stop_punctuations:
                sentence_length += 1
        sentence_length_average = (sentence_length_average * sentence_num + sentence_length) / (sentence_num + 1)
        sentence_num += 1

    pos_words = pos_tag(words)
    nouns, verbs, adjectives, adverbs = 0, 0, 0, 0
    for pos in pos_words:
        if pos[1] == 'NN' or pos[1] == 'NNP' or pos[1] == 'NNS' or pos[1] == 'NNPS':
            nouns += 1
        if pos[1] == 'JJ' or pos[1] == 'JJR' or pos[1] == 'JJS':
            adjectives += 1
        if pos[1] == 'RB' or pos[1] == 'RBR' or pos[1] == 'RBS':
            adverbs += 1
        if pos[1] == 'VB' or pos[1] == 'VBD' or pos[1] == 'VBG' or pos[1] == 'VBN' or pos[1] == 'VBP' \
                or pos[1] == 'VBZ':
            verbs += 1
    distribution = {
        'punctuation to word token': punctuation_tokens / word_tokens,
        'uppercase to word token': uppercase_tokens / word_tokens,
        'stop to word token': stop_word_tokens / word_tokens,
        'word type to token': len(vocabulary) / (stop_word_tokens + word_tokens),
        'noun': nouns / word_tokens,
        'adjective': adjectives / word_tokens,
        'verb': verbs / word_tokens,
        'adverb': adverbs / word_tokens,
        'sentence length': sentence_length_average
    }
    return distribution


def get_text_distribution(text, verbose=False):
    sentences = sent_tokenize(text)
    merged_sentences = []
    total_length = 0
    counted_sentences = []
    for sentence in sentences:
        sentence_length = 0
        for word in word_tokenize(sentence):
            if word not in punctuations and word not in stop_punctuations:
                sentence_length += 1
        if total_length == 0 or total_length + sentence_length < 50:
            counted_sentences.append(sentence)
        else:
            merged_sentences.append(sentences_to_text(counted_sentences))
            counted_sentences.clear()
            total_length = 0
            counted_sentences.append(sentence)
        total_length += sentence_length
    if len(merged_sentences) == 0 and len(counted_sentences) != 0:
        merged_sentences.append(sentences_to_text(counted_sentences))

    distributions = []
    for sample in merged_sentences:
        distribution = get_metrics(sample)
        distributions.append(distribution)
    for key in list(distributions[0].keys()):
        samples_added = 0
        distribution_sum = 0
        for index in range(len(distributions)):
            distribution = distributions[index]
            distribution_sum += distribution[key]
            samples_added += 1
        distributions[-1][key] = distribution_sum / samples_added
    final_distribution = distributions[-1]
    if verbose:
        print("Average Text Metrics")
        print("Punctuation to word token count ratio\t\t\t\t\t" + "{0:.2f}".format(final_distribution['punctuation to word token']))
        print("Uppercase word tokens to all word tokens count ratio\t" + "{0:.2f}".format(final_distribution['uppercase to word token']))
        print("Stop words to word token count ratio\t\t\t\t\t" + "{0:.2f}".format(final_distribution['stop to word token']))
        print("Word type to token ratio\t\t\t\t\t\t\t\t" + "{0:.2f}".format(final_distribution['word type to token']))
        print("Number of nouns\t\t\t\t\t\t\t\t\t\t\t" + "{0:.2f}".format(final_distribution['noun']))
        print("Number of adjectives\t\t\t\t\t\t\t\t\t" + "{0:.2f}".format(final_distribution['adjective']))
        print("Number of verbs\t\t\t\t\t\t\t\t\t\t\t" + "{0:.2f}".format(final_distribution['verb']))
        print("Number of adverbs\t\t\t\t\t\t\t\t\t\t" + "{0:.2f}".format(final_distribution['adverb']) + "\n")
    return final_distribution


def get_corpus_distribution(directory, verbose=False):
    sample_list = listdir(directory)
    distributions = []
    for sample in sample_list:
        if verbose:
            print(str(sample[:-4] + " Metrics"))
        sample_file = directory + sample
        with open(sample_file, encoding="utf8") as open_file:
            data = open_file.read()
        distribution = get_text_distribution(data, verbose=verbose)
        distributions.append(distribution)
        if verbose:
            print("Punctuation to word token count ratio\t\t\t\t\t" + "{0:.2f}".format(
                distribution['punctuation to word token']))
            print("Uppercase word tokens to all word tokens count ratio\t" + "{0:.2f}".format(
                distribution['uppercase to word token']))
            print("Stop words to word token count ratio\t\t\t\t\t" + "{0:.2f}".format(
                distribution['stop to word token']))
            print("Word type to token ratio\t\t\t\t\t\t\t\t" + "{0:.2f}".format(distribution['word type to token']))
            print("Number of nouns\t\t\t\t\t\t\t\t\t\t\t" + "{0:.2f}".format(distribution['noun']))
            print("Number of adjectives\t\t\t\t\t\t\t\t\t" + "{0:.2f}".format(distribution['adjective']))
            print("Number of verbs\t\t\t\t\t\t\t\t\t\t\t" + "{0:.2f}".format(distribution['verb']))
            print("Number of adverbs\t\t\t\t\t\t\t\t\t\t" + "{0:.2f}".format(distribution['adverb']) + "\n")
    for key in list(distributions[0].keys()):
        samples_added = 0
        distribution_sum = 0
        for index in range(len(distributions)):
            distribution = distributions[index]
            distribution_sum += distribution[key]
            samples_added += 1
        distributions[-1][key] = distribution_sum / samples_added
    final_distribution = distributions[-1]
    if verbose:
        print("Average Corpus Metrics")
        print("Punctuation to word token count ratio\t\t\t\t\t" + "{0:.2f}".format(final_distribution['punctuation to word token']))
        print("Uppercase word tokens to all word tokens count ratio\t" + "{0:.2f}".format(final_distribution['uppercase to word token']))
        print("Stop words to word token count ratio\t\t\t\t\t" + "{0:.2f}".format(final_distribution['stop to word token']))
        print("Word type to token ratio\t\t\t\t\t\t\t\t" + "{0:.2f}".format(final_distribution['word type to token']))
        print("Number of nouns\t\t\t\t\t\t\t\t\t\t\t" + "{0:.2f}".format(final_distribution['noun']))
        print("Number of adjectives\t\t\t\t\t\t\t\t\t" + "{0:.2f}".format(final_distribution['adjective']))
        print("Number of verbs\t\t\t\t\t\t\t\t\t\t\t" + "{0:.2f}".format(final_distribution['verb']))
        print("Number of adverbs\t\t\t\t\t\t\t\t\t\t" + "{0:.2f}".format(final_distribution['adverb']) + "\n")
    return final_distribution


def castro_masking(text: str):
    masked_text = change_contraction_distribution(text)
    if masked_text == text:
        text = remove_parenthetical_phrases(text)
        text = remove_appositive_phrases(text)
        text = remove_discourse_markers(text)
    else:
        text = masked_text

    return synonym_substitution(text)


def mihaylova_masking(input_text: str):
    def obfuscate_all(original_text, text_parts, helpers):
        document_stats = DocumentStats(original_text, helpers['spellcheck'], helpers['stopwords'])

        for text in text_parts:
            text.obfuscated_text = obfuscate_text(text.original_text, document_stats, helpers)
        return text_parts

    def obfuscate_text(text, document_stats, helpers):
        # 1. Apply general transfomations
        text = apply_general_transformations(text, helpers)
        # 2. According to the document measures - apply transformations to get closer to the averages
        text = apply_obfuscation(text, document_stats, helpers)
        # 3. Add noise to the text
        text = add_noise(text, helpers)

        # Add an empty space at the end of the text to make meaningful text when concatenating the output
        text = text + ' '

        return text

    def apply_general_transformations(text, helpers):
        text = apply(text)

        text = transform_equation(text)

        text = replace_numbers(text)

        text = replace_short_forms(text)

        symbolReplacement = helpers['symbolReplacement']
        text = symbolReplacement.replace_symbols(text)

        return text

    def apply_obfuscation(text, document_stats, helpers):
        spellchecker = helpers['spellcheck']
        stopwords = helpers['stopwords']
        errorCreator = helpers['errorCreator']
        punctuation = helpers['punctuation']
        fillerWords = helpers['fillerWords']
        paraphraseCorpus = helpers['paraphraseCorpus']

        spell_correction = True
        obfuscate_stop_words = True
        obfuscate_punctuation = True
        transform_sentences = True
        change_pos_count = False  # This does not work very well, especially when sentences are transformed.
        substitute_words = True
        uppercase = True
        paraphrase_corpus = True

        # Obfuscate uppercase words
        if uppercase:
            text = obfuscate_uppercase(text)

        # Paraphrasing
        if paraphrase_corpus:
            text = paraphraseCorpus.obfuscate(text)

        # Obfuscate stop words
        if obfuscate_stop_words:
            text = obfuscate_stopwords(text, document_stats, stopwords)

        # Obfuscate punctuation
        if obfuscate_punctuation:
            text = obfuscate_punctuation_count(text, document_stats, punctuation)

        # Remove adjectives
        if change_pos_count:
            # TODO -Georgi: We don't use this for now, but we should re-do this to be more effective
            text = obfuscate_pos_count(text, document_stats)

        # Substitute some words with their synonims, hypernims or definitions
        if substitute_words:
            text = obfuscate_unique_words_count(text, document_stats)

        # Split or merge sentences
        if transform_sentences:
            text = obfuscate_sentence_length(text, document_stats, fillerWords)

        # Use spellchecker
        if spell_correction:
            text = obfuscate_spelling(text, document_stats, spellchecker, errorCreator)

        return text

    def add_noise(text, helpers):
        apply_translation = False  # Can be included later.
        british_american = True
        filler_words = True

        # Applying two-way translation obfuscation on the text
        if apply_translation:
            translator = Translation()
            languages = [Language.Croatian, Language.Estonian]
            text = translator.translate(text, languages)

        if british_american:
            britishToAmerican = helpers['britishToAmerican']
            text = britishToAmerican.create_errors(text)

        if filler_words and random.choice([False, False, True, False, False]):
            fillerWords = helpers['fillerWords']
            text = fillerWords.insert_random(text)

        return text

    def split_text(text):
        # text = text.strip() #Probably this was the problem, with the slight length bug.
        sentences = sent_tokenize(text)

        positioned_sentences = []
        pos = 0
        obfuscation_id = 1
        temp_sequence = ''

        for index, sentence in enumerate(sentences):
            sentence_length = len(sentence)

            if not temp_sequence:
                temp_sequence = sentence
            else:
                if len(sentences) == index + 1:
                    temp_sequence = temp_sequence + ' ' + sentence
                if word_count(temp_sequence + sentence) > MAX_SEQUENCE_LENGTH or len(sentences) == index + 1:
                    # Adding the current sequence would exceed the allowed length =>
                    # 1. Add the current sequence to the list
                    # 2. Reset the sequence with the new sentence
                    start_pos = pos
                    end_pos = start_pos + len(temp_sequence) + 1  # Add one position for interval
                    # Clear the new lines from the text
                    temp_sequence = temp_sequence.strip().replace('\n', ' ')
                    temp_sequence = temp_sequence + ' '  # Add an interval at the end of the sentence
                    text_part = TextPart(obfuscation_id, start_pos, end_pos, temp_sequence)
                    positioned_sentences.append(text_part)
                    pos = end_pos + 1
                    obfuscation_id = obfuscation_id + 1

                    # The sentence is starting the new sequence
                    temp_sequence = sentence
                else:
                    # Still allowed length - add the sentence to the temp sequence and continue
                    if temp_sequence:
                        temp_sequence = temp_sequence + ' ' + sentence
                    else:
                        temp_sequence = sentence

        return positioned_sentences

    helpers = {
        'stopwords': StopWords(),
        'spellcheck': SpellCheck(),
        'errorCreator': ErrorCreator(),
        'punctuation': Punctuation(),
        'britishToAmerican': BritishAmericanNormalization(),
        'fillerWords': FillerWords(),
        'symbolReplacement': SymbolReplacement(),
        'paraphraseCorpus': ParaphraseCorpus()
    }

    file_parts = split_text(input_text)
    file_parts = obfuscate_all(input_text, file_parts, helpers)

    output_text = ""
    text_parts = file_parts
    for obfuscation in text_parts:
        output_text += obfuscation.obfuscated_text
    return output_text

def rahgouy_masking(original_text: str, author_texts:str=None):
    def get_syns_word2vec(w, pos, get_wordnet_pos):
        if w in lm.vocab:
            words = lm.most_similar(w, topn=100)
            syn, syn_words = {}, []
            for word in words:
                if (not ".com" in word[0]) and (not "_" in word[0]):
                    cleaned_word = word[0].replace("#", "")
                    cleaned_word = cleaned_word.replace("-", "")
                    cleaned_word = cleaned_word.replace(".", "")
                    cleaned_word = cleaned_word.replace("=", "")
                    cleaned_word = cleaned_word.replace(" ", "")
                    cleaned_word = cleaned_word.replace("/", " ")
                    cleaned_word = cleaned_word.replace("\\", " ")
                    if len(cleaned_word) > 1:
                        syn[cleaned_word.lower()] = word[1]
                        syn_words.append(cleaned_word.lower())
            tagged_words = nltk.pos_tag(syn_words)
            s = {}
            for tagged_word in tagged_words:
                if tagged_word[1].startswith(pos[0]):
                    s[lemmatizer.lemmatize(tagged_word[0], get_wordnet_pos(pos[0]))] = syn[tagged_word[0]]
            synsets = [[syn, s[syn]] for syn in s if syn.lower() != w.lower()]
            return synsets
        return []

    def get_syns_wordnet(w, pos, get_wordnet_pos):
        p = get_wordnet_pos(pos)
        if p is None:
            return []
        synset = wn.synsets(w, p)
        if len(synset) == 0:
            return []
        words = [word.lemma_names() for word in synset]
        words = [ww for word in words for ww in word
                 if ww.lower() != w.lower() and lemmatizer.lemmatize(ww, p) != lemmatizer.lemmatize(w, p)]
        words = {words[i]: 0 for i in range(0, len(words))}
        syn_words = [item for item in words]
        word = wn.synsets(w)[0]
        syn = []
        for syn_word in syn_words:
            synset = wn.synsets(syn_word)
            similarity = wn.wup_similarity(word, synset[0])
            if similarity is None:
                similarity = 0
            syn.append([syn_word, similarity])
        syn = sorted(syn, key=lambda x: x[1], reverse=True)
        return [[x[0].replace("_", " "), x[1]] for x in syn]

    synonym_default = "wordnet"
    syns = {"wordnet":get_syns_wordnet , "word2vec":get_syns_word2vec}
    lm = gensim.models.KeyedVectors.load_word2vec_format(dirname(dirname(abspath(__file__))) + "/word2vec_model.bin", binary = True , limit = 600000)
    lemmatizer = nltk.stem.WordNetLemmatizer()
    synonyms = syns[synonym_default]
    with codecs.open(dirname(dirname(abspath(__file__))) + '/phrases', 'r', 'utf-8', 'ignore') as f:
        text = f.read()
    phrases = [tuple( w.lower() for w in word.split( ' ' ) ) for word in text.split( '\n' )]
    tokenizer = MWETokenizer( phrases )
    with codecs.open(dirname(dirname(abspath(__file__))) + "/phrase_synonyms.json", "r", "utf-8" ) as JSON:
        phrase_syns = json.load( JSON )
    with codecs.open(dirname(dirname(abspath(__file__))) + "/full_format_contractions.json" , "r" , "utf-8") as JSON:
        full_format_contractions = json.load(JSON)
    with codecs.open(dirname(dirname(abspath(__file__))) + "/short_format_contractions.json", "r", "utf-8" ) as JSON:
        short_format_contractions = json.load( JSON )

    if author_texts is None:
        with open(dirname(dirname(abspath(__file__))) + '/Dictionaries/big.txt', 'r') as content_file:
            author_texts = content_file.read()

    def obfuscate(author_texts, parts):
        original_text = ' '.join([part["original"] for part in parts])
        original_dict = get_words_dict(original_text, get_wordnet_pos_JRV)
        author_dict = get_words_dict(author_texts, get_wordnet_pos_JRV)
        dict_args = {
            "original_dict": original_dict,
            "author_dict": author_dict
        }
        S, F = calculate_full_short_format(author_texts)
        _parts = obfuscate_contractions(author_texts, parts, S, F)
        obfuscated_words = []
        for part in _parts:
            obfuscate_sentence(part, dict_args, obfuscated_words)
        obj = approach_3(author_texts, _parts)
        return obj

    def calculate_full_short_format(author_text):
        full_format, short_format = 0, 0
        for sent in nltk.sent_tokenize(author_text):
            tokens = word_tokenizer(sent)
            for words in tokens:
                is_full_format = False
                is_short_format = False
                for word in full_format_contractions:
                    if word == words:
                        is_full_format = True
                for word in short_format_contractions:
                    if word == words:
                        is_short_format = True
                if is_full_format:
                    full_format = full_format + 1
                if is_short_format:
                    short_format = short_format + 1
        return short_format, full_format

    def obfuscate_contractions(author_text, parts, S, F):
        _parts = []
        for part in parts:
            tokens = word_tokenizer(part["obfuscation"])
            index = 0
            for words in tokens:
                if F > S:
                    for word in full_format_contractions:
                        if word.lower() == words.lower():
                            tokens[index] = full_format_contractions[word]
                else:
                    for word in short_format_contractions:
                        if word.lower() == words.lower():
                            if '/' not in short_format_contractions[word]:
                                tokens[index] = short_format_contractions[word]
                index = index + 1
            part["obfuscation"] = ' '.join(tokens)
            _parts.append(part)
        return _parts

    def obfuscate_sentence(part, dict_args, obfuscated_words):
        sentence = part["obfuscation"]
        sentence = do_initial_replacements(sentence)
        words = word_tokenizer(sentence)
        tagged_words = nltk.pos_tag(words)
        index = 0
        for tagged_word in tagged_words:
            if check_conditions(words, index, tagged_word[1]):
                pos = get_wordnet_pos_JRV(tagged_word[1])
                if (pos is not None):
                    if len(tagged_word[0].split(' ')) >= 2:
                        syns_1 = [phrase_syns[syn] for syn in phrase_syns if syn == tagged_word[0]]
                        syns_2 = synonyms(tagged_word[0], tagged_word[1], get_wordnet_pos_JRV)
                        syns = syns_1 + syns_2
                        if len(syns) != 0:
                            syns = [[syn, (get_score_diff(words, index, syn) +
                                           1 / (1 + get_word_freq(dict_args['author_dict'], syn, pos,
                                                                  get_wordnet_pos_JRV))) / 2]
                                    for syn in syns[0]]
                            syns = sorted(syns, key=lambda x: x[1], reverse=True)
                            is_good, candidate_index = get_best_replacement(obfuscated_words, words[index], syns)
                            if is_good:
                                obfuscated_words.append([words[index], syns[candidate_index][0]])
                                words[index] = get_adjust(syns[candidate_index][0], tagged_word[1])
                    else:
                        syns = synonyms(tagged_word[0], tagged_word[1], get_wordnet_pos_JRV)
                        if len(syns) != 0:
                            valid = True
                            freq_w = get_word_freq(dict_args["original_dict"], tagged_word[0], pos, get_wordnet_pos_JRV)
                            if checking_author_dict(dict_args["author_dict"], tagged_word[0]):
                                syns = [[syn[0], syn[1],
                                         get_word_freq(dict_args["author_dict"], syn[0], pos, get_wordnet_pos_JRV)] for
                                        syn in syns]
                                syns = [[syn[0], (1 / (1 + syn[2]) + syn[1]) / 2] for syn in syns if syn[2] <= freq_w]
                            elif freq_w >= 2:
                                syns = [[syn[0], syn[1],
                                         get_word_freq(dict_args["author_dict"], syn[0], pos, get_wordnet_pos_JRV)]
                                        for syn in syns]
                                syns = [[syn[0], (1 / (1 + syn[2]) + syn[1]) / 2] for syn in syns]
                            else:
                                valid = False
                            syns = sorted(syns, key=lambda x: x[1], reverse=True)
                            if len(syns) != 0 and valid:
                                syns = [[syn[0], (syn[1] + get_score_diff(words, index, syn[0])) / 2] for syn in syns]
                                syns = sorted(syns, key=lambda x: x[1], reverse=True)
                                is_good, candidate_index = get_best_replacement(obfuscated_words, words[index], syns)
                                if is_good:
                                    obfuscated_words.append([words[index], syns[candidate_index][0]])
                                    words[index] = get_adjust(syns[candidate_index][0], tagged_word[1])
            index = index + 1
        part["obfuscation"] = ' '.join(words)

    def do_initial_replacements(sentence):
        sent = sentence
        if "so" in sent:
            words = word_tokenizer(sent)
            tagged_words = nltk.pos_tag(words)
            for i in range(0, len(tagged_words)):
                if tagged_words[i][0].lower() == "so":
                    if (i + 1) < len(tagged_words):
                        if tagged_words[i + 1][1].startswith('J'):  # or tagged_words[i+1][1].startswith('R'):
                            words[i] = "very"
            sent = ' '.join(words)
        sent = sent.replace("may be", "maybe")
        return sent

    def check_conditions(words, index, _pos):
        words_list = ("am", "is", "are", "have", "has", "was", "were"
                      , "did", "does", "can", "could", "not", "will")
        # I deleted the 'do' because our tokenizer splite them if we got 'do not' word
        # other time we allowed to use them as well as
        word = words[index].lower()
        if (index + 1) != len(words) and words[index + 1] == "n't":
            return False
        if word in words_list:
            return False
        if word in string.punctuation:
            return False
        return True

    def checking_author_dict(author_dict, word):
        for w in author_dict:
            if word.lower() == w['word'][0].lower():
                return True
        return False

    def get_adjust(word, pos):
        # if pos == "NNS":
        #    return pattern.pluralize(pattern.singularize(word))
        if pos == "VB":
            return pattern.conjugate(word, 'inf')
        if pos == "VBP":
            return pattern.conjugate(word, '2sg')
        if pos == "VBZ":
            return pattern.conjugate(word, '3sg')
        if pos == "VBG":
            return pattern.conjugate(word, 'part')
        if pos == "VBD":
            return pattern.conjugate(word, '2sgp')
        if pos == "VBN":
            return pattern.conjugate(word, 'ppart')
        return word

    def approach_3(author_texts, parts):
        original_text = ' '.join([part["original"] for part in parts])
        mean, average = get_text_mean(author_texts)
        short, long = short_long_calculater(original_text, mean)
        if short == long:
            return parts
        new_parts = []
        obfuscation_id = 1
        p = 0
        while p < len(parts):
            if len(word_tokenizer(parts[p]["obfuscation"])) <= mean and short > long:
                if (p + 1) < len(parts):
                    sent = parts[p]["original"] + parts[p + 1]["original"]
                    sent_words = sent.split(' ')
                    if len(sent_words) <= 50:
                        part = {
                            "obfuscation": parts[p]["obfuscation"][:(len(parts[p]["obfuscation"])) - 1]
                                           + " and " + parts[p + 1]["obfuscation"],
                            "original-start-charpos": parts[p]["original-start-charpos"],
                            "original-end-charpos": parts[p]["original-end-charpos"]
                                                    + len(parts[p + 1]["original"]),
                            "original": parts[p]["original"] + parts[p + 1]["original"],
                            "obfuscation-id": obfuscation_id
                        }
                        new_parts.append(part)
                        p += 1
                    else:
                        parts[p]["obfuscation-id"] = obfuscation_id
                        new_parts.append(parts[p])
                else:
                    parts[p]["obfuscation-id"] = obfuscation_id
                    new_parts.append(parts[p])
            elif len(word_tokenizer(parts[p]["obfuscation"])) > mean and short < long:
                words = word_tokenizer(parts[p]["obfuscation"])
                tagged_words = nltk.pos_tag(words)
                is_divided = False
                for ind in range(0, len(tagged_words)):
                    if tagged_words[ind][1].startswith('CC') and (not is_divided):
                        is_divided = True
                        # sent = get_sentence(words[:ind]) #' '.join(words[:ind])
                        sent = ' '.join(words[:ind])
                        part_1 = {
                            "obfuscation": sent + ".",
                            "original-start-charpos": parts[p]["original-start-charpos"],
                            "original-end-charpos": parts[p]["original-start-charpos"]
                                                    + (len(sent)),
                            "original": parts[p]["original"][:len(sent)],  # len(sent)+1
                            "obfuscation-id": obfuscation_id
                        }
                        obfuscation_id += 1
                        new_parts.append(part_1)
                        # sent_2 = get_sentence(words[ind+1:])#' '.join(words[ind + 1 : ])
                        sent_2 = ' '.join(words[ind:])  # ind+1:
                        part_2 = {
                            "obfuscation": sent_2,
                            "original-start-charpos": part_1["original-end-charpos"] + 1,
                            "original-end-charpos": parts[p]["original-end-charpos"],
                            "original": parts[p]["original"][len(sent):],  # len(sent)+1
                            "obfuscation-id": obfuscation_id
                        }
                        # obfuscation_id += 1
                        new_parts.append(part_2)
                if not is_divided:
                    parts[p]["obfuscation-id"] = obfuscation_id
                    new_parts.append(parts[p])
            else:
                parts[p]["obfuscation-id"] = obfuscation_id
                new_parts.append(parts[p])
            p += 1
            obfuscation_id += 1
        return new_parts

    def get_parts(problem):
        obfuscation_id = 1
        sentences = nltk.sent_tokenize(problem)
        curr = 0
        prev = 0
        parts = []
        for s in sentences:
            curr = problem.find(s, curr)
            assert curr > -1
            pad = problem[prev:curr]
            part = {
                "original": pad + s,
                "original-start-charpos": prev,
                "original-end-charpos": curr + len(s) - 1,
                "obfuscation": s,
                "obfuscation-id": obfuscation_id
            }
            parts.append(part)
            prev = curr + len(s)
            curr = curr + 1
            obfuscation_id = obfuscation_id + 1
        return parts

    def get_words_dict(text, get_wordnet_pos):
        words = word_tokenizer(text)
        tagged_words = nltk.pos_tag(words)
        words = []
        for tagged_word in tagged_words:
            pos = get_wordnet_pos(tagged_word[1])
            if pos is not None:
                words.append((lemmatizer.lemmatize(tagged_word[0].lower(), pos), tagged_word[1]))
        dist = nltk.FreqDist(words)
        words_dict = []
        for word, frequency in dist.most_common(len(tagged_words)):
            word_dict = {}
            word_dict['word'] = word
            word_dict['count'] = frequency
            words_dict.append(word_dict)
        return words_dict

    def get_best_replacement(obfuscated_words, word, candidate_list):
        index = 0
        if len(obfuscated_words) != 0:
            while True:
                is_good = True
                for words in obfuscated_words:
                    # if words[0] == word:
                    if words[1] == candidate_list[index][0]:
                        is_good = False
                if is_good:
                    return True, index
                else:
                    index = index + 1
                    if index == len(candidate_list):
                        return False, index
        else:
            return True, index

    def get_text_mean(text):

        sent_lengths = [len(word_tokenizer(sent)) for sent in nltk.sent_tokenize(text)]
        sent_lengths = sorted(sent_lengths)
        if len(sent_lengths) % 2 == 0 and len(sent_lengths) > 0:
            mean = (sent_lengths[int(len(sent_lengths) / 2)] + sent_lengths[int(len(sent_lengths) / 2) + 1]) / 2
        elif len(sent_lengths) > 0:
            mean = sent_lengths[int(len(sent_lengths) / 2)]
        else:
            print()
            raise Exception("There are no author texts")
        average = len(sent_lengths) / sum(sent_lengths)
        return mean, average

    def short_long_calculater(text, mean):
        short, long = 0, 0
        sent_lengths = [len(word_tokenizer(sent)) for sent in nltk.sent_tokenize(text)]
        for length in sent_lengths:
            if length <= mean:
                short += 1
            else:
                long += 1
        return short, long

    def get_wordnet_pos_JRV(treebank_tag):
        if treebank_tag.startswith('J'):
            return wn.ADJ
        elif treebank_tag.startswith('R'):
            return wn.ADV
        elif treebank_tag.startswith('V'):
            return wn.VERB
        else:
            return None

    def get_word_freq(dict, word, pos, get_wordnet_pos):
        for w in dict:
            if (w["word"][0] == lemmatizer.lemmatize(word.lower(), pos)) and (get_wordnet_pos(w["word"][1]) == pos):
                return w["count"]
        return 0

    def get_score_diff(words, i, new_word):
        orig = ' '.join(words)
        w = words[i]
        words[i] = new_word
        new = ' '.join(words)
        words[i] = w
        return lm.wmdistance(orig, new)

    def word_tokenizer(sentence):
        regex_tokenizer = RegexpTokenizer("\s+", gaps=True)
        words = tokenizer.tokenize(regex_tokenizer.tokenize(sentence))
        ws = []
        for word in words:
            if word[len(word) - 1] in string.punctuation:
                ws.append(word[:len(word) - 1])
                ws.append(word[len(word) - 1])
            else:
                ws.append(word)
        return [w.replace("_", " ") for w in ws if len(w) != 0]

    parts = get_parts(original_text)
    obj = obfuscate(author_texts, parts)

    output_text = ""
    text_parts = obj
    for obfuscation in text_parts:
        output_text += obfuscation['obfuscation']
    return output_text
