# -*- coding: cp1252 -*-

from os import listdir, mkdir
from os import remove
from os.path import isdir, dirname, abspath, isfile
from shutil import copy, rmtree, copytree
from subprocess import call
from time import localtime, strftime
from time import time
from itertools import combinations
from os import rename, path
from json import dump
from numpy import mean, shape
from collections import Counter

from Code.Masking_Operations import synonym_substitution, change_contraction_distribution, \
    change_sentence_distribution, remove_parenthetical_phrases, remove_appositive_phrases, remove_discourse_markers
from Code.Utils import fix_encoding, get_answers, get_answers_accuracy, \
    fitness_function, get_obfuscation_mihaylova, get_obfuscation_rahgouy, \
    cdaa_preprocessing, author_attribution_preprocessing, author_verification_preprocessing, get_truth


def allred_masking(dataset_directory, verbose=False, output_directory=None, remove_directory=True):
    if output_directory is None:
        author_masking = dirname(dirname(abspath(__file__))) + '/AMT_Results'
        output_directory = author_masking + '/Allred'

    epsilon = 0.00
    replacement_rate = 0.20
    replacement_loop_stop = 20

    error_file = dirname(dirname(abspath(__file__))) + "/Code/Allred/error.txt"
    directory = dirname(dirname(abspath(__file__))) + "/Code/Allred/"
    language = dirname(dirname(abspath(__file__))) + "/Code/Allred/language.txt"
    original_file = dirname(dirname(abspath(__file__))) + "/Code/Allred/original.txt"

    if isdir(output_directory) and remove_directory:
        rmtree(output_directory)

    paths = listdir(dataset_directory)

    for path in paths:
        if isdir(dataset_directory + '/' + path):
            if not isdir(output_directory + '/' + path):
                mkdir(output_directory + '/' + path)
            paths.remove(path)
            new_paths = listdir(dataset_directory + '/' + path)
            for new_path in new_paths:
                paths.append(path + '/' + new_path)

    for path in paths:
        while True:
            error = 100
            print(strftime("%H:%M:%S", localtime()), "working on ", path)
            if isfile(original_file):
                remove(original_file)
            if isfile(error_file):
                remove(error_file)
            if isfile(dirname(dirname(abspath(__file__))) + "/Code/Allred/gallegan.txt"):
                remove(dirname(dirname(abspath(__file__))) + "/Code/Allred/gallegan.txt")
            if isfile(dirname(dirname(abspath(__file__))) + "/Code/Allred/catalan.txt"):
                remove(dirname(dirname(abspath(__file__))) + "/Code/Allred/catalan.txt")
            if isfile(dirname(dirname(abspath(__file__))) + "/Code/Allred/spanish.txt"):
                remove(dirname(dirname(abspath(__file__))) + "/Code/Allred/spanish.txt")
            if isfile(dirname(dirname(abspath(__file__))) + "/Code/Allred/wordnet.txt"):
                remove(dirname(dirname(abspath(__file__))) + "/Code/Allred/wordnet.txt")

            from Code.WordNet import run as run_wordnet
            run_wordnet(dataset_directory + '/' + path, replacement_rate, replacement_loop_stop)

            from Code.Feature_Extractor import run as run_feature_extractor
            run_feature_extractor('wordnet', 1 if verbose else 0, int(path.split('_')[0][-1]))

            copy(dataset_directory + '/' + path, directory + '/input.txt')

            from Code.apertium_linux import run_catalan
            run_catalan()
            run_feature_extractor('catalan', 1 if verbose else 0, int(path.split('_')[0][-1]))

            from Code.apertium_linux import run_spanish
            run_spanish()
            run_feature_extractor('spanish', 1 if verbose else 0, int(path.split('_')[0][-1]))

            from Code.apertium_linux import run_gallegan
            run_gallegan()
            run_feature_extractor('gallegan', 1 if verbose else 0, int(path.split('_')[0][-1]))

            with open(error_file, 'r') as open_file:
                error_file = open_file.read().split(',')[0]
                best_error = float(open_file.read().split(',')[1])
            if not best_error < error:
                break
            else:
                with open(dirname(dirname(abspath(__file__))) + "/Code/Allred/" + error_file + '.txt', 'w+') as error_file:
                    output = error_file.read()
        with open(output_directory + '/' + path, 'w+') as output_file:
            output_file.write(output)



def castro_masking(input_directory, verbose=False, output_directory=None, remove_directory=True):
    input_files = listdir(input_directory)

    if output_directory is None:
        author_masking = dirname(dirname(abspath(__file__))) + '/AMT_Results'
        output_directory = author_masking + '/Castro'

    if remove_directory and isdir(output_directory):
        rmtree(output_directory)

    if not isdir(output_directory):
        try:
            mkdir(output_directory)
        except FileExistsError:
            pass

    for file in input_files:
        if verbose:
            print(strftime("%H:%M:%S", localtime()), "working on ", file)

        with open(input_directory + "/" + file, 'r', encoding='utf8', errors='ignore') as input_file:
            text = input_file.read()

        masked_text = change_contraction_distribution(text)
        if masked_text == text:
            text = remove_parenthetical_phrases(text)
            text = remove_appositive_phrases(text)
            text = remove_discourse_markers(text)
        else:
            text = masked_text

        text = synonym_substitution(text)

        with open(output_directory + "/" + file, 'w+', encoding='utf8', errors='ignore') as output_file:
            output_file.write(text)


def mihaylova_masking(input_directory, output_directory=None, remove_directory=True):
    if output_directory is None:
        author_masking = dirname(dirname(abspath(__file__))) + '/AMT_Results'
        output_directory = author_masking + '/Mihaylova'

    try:
        mkdir(output_directory)
    except FileExistsError:
        pass
    input_files = listdir(input_directory)

    for file in input_files:
        mkdir(input_directory + "/" + file[:-4])
        copy(input_directory + "/" + file, input_directory + "/" + file[:-4] + '/original.txt')
        remove(input_directory + "/" + file)

    arguments = ["python3",
                 dirname(dirname(abspath(__file__))) +
                 "/AMT/Mihaylova/AuthorObfuscation/AuthorObfuscation/AuthorObfuscation.py",
                 "-i", input_directory, "-o", output_directory]
    call(arguments)

    return output_directory


def rahgouy_preprocessing(input_directory, known_directory):
    for file in listdir(input_directory):
        if file.__contains__('.txt'):
            with open(input_directory + '/' + file, 'r', encoding='utf8', errors='ignore') as read_file:
                data = read_file.read()
            while data.__contains__('\n\n'):
                data = data.replace('\n\n', '\n')
            with open(input_directory + '/' + file, 'w', encoding='utf8') as write_file:
                write_file.write(data)


def rahgouy_masking(input_directory, known_directory, output_directory=None, remove_directory=True):
    pretrained_model = dirname(dirname(abspath(__file__))) + '/AMT/Rahgouy/author_obf_pan2018' \
                                                             '/trained_model.gz'

    if output_directory is None:
        author_masking = dirname(dirname(abspath(__file__))) + '/AMT_Results'
        output_directory = author_masking + '/Rahgouy'

    try:
        mkdir(output_directory)
    except FileExistsError:
        pass
    input_files = listdir(input_directory)

    for file in input_files:
        mkdir(input_directory + "/" + file[:-4])
        copy(input_directory + "/" + file, input_directory + "/" + file[:-4] + '/original.txt')
        remove(input_directory + "/" + file)

    author_files = listdir(known_directory)

    for file in author_files:
        for search in listdir(input_directory):
            if search.__contains__(file[:-6]):
                copy(known_directory + "/" + file, input_directory + "/" + search + '/' + file[-5] + ".txt")

    arguments = ["python3",
                 dirname(dirname(abspath(__file__))) +
                 "/AMT/Rahgouy/author_obf_pan2018/"
                 "obfuscation_script.py", "-i", input_directory, "-o", output_directory,
                 '-lm', pretrained_model, '-t', 'true']

    call(arguments)

    return output_directory


def keswani_masking(text):
    # text = keswani_ilt(text)
    return text


def bakhteev_masking(text):
    text = synonym_substitution(text)
    text = change_contraction_distribution(text)
    text = change_sentence_distribution(text)
    # text = lstm_encoder_decoder(text)
    # text = change_introductory_phrases()
    return text


def kocher_masking(known_directory: str, unknown_directory: str, output_directory: str, input_directory=None, remove_directory=True):
    if remove_directory and isdir(output_directory):
        rmtree(output_directory)

    if input_directory is None:
        input_directory = dirname(dirname(abspath(__file__))) + \
                          "/AMT_Input/"

    if not isdir(output_directory):
        mkdir(output_directory)


    if isdir(input_directory):
        rmtree(input_directory)
    mkdir(input_directory)

    known_files = listdir(known_directory)
    unknown_files = listdir(unknown_directory)


    for file in known_files:
        if file.startswith('.'):
            continue
        if not isdir(input_directory + file.split('_')[0]):
            mkdir(input_directory + file.split('_')[0])
        copy(known_directory + '/' + file, input_directory + file.split('_')[0] + '/' + file)

    for file in unknown_files:
        if file.startswith('.'):
            continue
        if not isdir(input_directory + file.split('_')[0]):
            mkdir(input_directory + file.split('_')[0])
        copy(unknown_directory + '/' + file, input_directory + file.split('_')[0] + '/original.txt')

    arguments = ["python2",
                 dirname(dirname(abspath(__file__))) +
                 "/AMT/Kocher/panAM.py",
                 "-i", input_directory, "-o", output_directory]
    call(arguments)

    return output_directory

def lsvm_attribution(input_directory=None, output_directory=None):
    from numpy import array
    from numpy import concatenate
    from numpy import max, min, arange
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.preprocessing import normalize
    from sklearn.svm import LinearSVC

    if input_directory is None:
        input_directory = dirname(dirname(abspath(__file__))) + \
                          "/AAS_Input"

    def extract_features(directory_name: str, vectorizer=None):
        def character_unigram(file_name: str):
            unigram_list = [0] * 95
            with open(file_name, errors='ignore') as open_file:
                file_data = open_file.read()
            for character in file_data:
                code = ord(character) - 32
                if -1 < code < 95:
                    unigram_list[code] += 1

            unigram_list = array(unigram_list)
            unigram_list = unigram_list.reshape(1, -1)
            unigram_list = normalize(unigram_list)
            unigram_list = unigram_list[0]

            return unigram_list

        def character_bigram(file_name: str):
            bigram_list = [0] * 95 ** 2
            with open(file_name, errors='ignore') as open_file:
                file_data = open_file.read()
            for index in range(len(file_data)):
                if index < (len(file_data) - 1):
                    code_1 = ord(file_data[index]) - 32
                    code_2 = ord(file_data[index + 1]) - 32
                    if -1 < code_1 < 95 and -1 < code_2 < 95:
                        bigram_list[(95 * code_1) + code_2] += 1
                else:
                    break

            bigram_list = array(bigram_list)
            bigram_list = bigram_list.reshape(1, -1)
            bigram_list = normalize(bigram_list)
            bigram_list = bigram_list[0]

            return bigram_list

        def bag_of_words(corpus: list):
            vectorizer = CountVectorizer()
            model = vectorizer.fit_transform(corpus)
            global corpus_vectorizer
            corpus_vectorizer = vectorizer
            vocabulary = model.toarray()
            vocabulary = normalize(vocabulary)
            return vocabulary, vectorizer

        def files_to_corpus(directory_name: str, preprocessed: bool):
            file_list = listdir(directory_name)
            corpus = []

            if not preprocessed:
                for file in file_list:
                    file_path = directory_name + '/' + file
                    with open(file_path, errors='ignore') as open_file:
                        file_data = open_file.read()
                    corpus.append(file_data)
            else:
                for directory in file_list:
                    if not directory.__contains__('candidate'):
                        continue
                    for file in listdir(directory_name + '/' + directory):
                        file_path = directory_name + '/' + directory + '/' + file
                        with open(file_path, errors='ignore') as open_file:
                            file_data = open_file.read()
                        corpus.append(file_data)

            return corpus

        def feature_vocabulary(feature: list, corpus_vectorizer):
            vectorizer = CountVectorizer(vocabulary=corpus_vectorizer.vocabulary_)
            model = vectorizer.fit_transform(feature)
            vocabulary = model.toarray()
            vocabulary = normalize(vocabulary)

            return vocabulary

        unigram_vector, bigram_vector = [], []
        file_list = listdir(directory_name)


        authors_list = []

        preprocessed = False
        for file in file_list:
            if isdir(directory_name + '/' + file):
                preprocessed = True
                break

        if not preprocessed:
            for file in file_list:
                if file.__contains__('.txt'):
                    file_path = directory_name + '/' + file
                    authors_list.append(file.split('_')[0])
                    unigram_vector.append(character_unigram(file_path))
                    bigram_vector.append(character_bigram(file_path))
        else:
            for directory in file_list:
                if not directory.__contains__('candidate'):
                    continue
                for file in listdir(directory_name + '/' + directory):
                    if file.__contains__('.txt'):
                        file_path = directory_name + '/' + directory + '/' + file
                        authors_list.append(directory)
                        unigram_vector.append(character_unigram(file_path))
                        bigram_vector.append(character_bigram(file_path))

        corpus = files_to_corpus(directory_name, preprocessed)
        if vectorizer is not None:
            vocabulary_vector = feature_vocabulary(corpus, vectorizer)
        else:
            vocabulary_vector, vectorizer = bag_of_words(corpus)

        feature_vector_3 = [0] * len(unigram_vector)
        feature_vector_7 = [0] * len(unigram_vector)

        for index in range(len(feature_vector_7)):
            feature_vector_3[index] = concatenate(
                (unigram_vector[index], bigram_vector[index], vocabulary_vector[index]))
            feature_vector_7[index] = vocabulary_vector[index]

        return array(feature_vector_3), array(feature_vector_7), array(authors_list), vectorizer

    v3, garbage, authors_list, vectorizer = extract_features(input_directory)
    test_feature, garbage, garbage, garbage = extract_features(input_directory + '/unknown', vectorizer)

    train_labels, train_data = authors_list, v3
    model = LinearSVC()
    model.fit(train_data, train_labels)
    predictions = model.predict(test_feature)
    decision_function = model.decision_function(test_feature)

    max_value = 1
    min_value = 0

    for decision_index in range(len(decision_function)):
        decision = decision_function[decision_index]
        min_element = min(decision)
        for element in range(len(decision)):
            decision_function[decision_index][element] = decision[element] - min_element
        decision = decision_function[decision_index]
        sum_total = sum(decision)
        for element in range(len(decision)):
            decision_function[decision_index][element] = decision[element] / sum_total
        decision = decision_function[decision_index]
    answers = []
    texts = listdir(input_directory + '/unknown')
    for i in range(len(texts)):
        answers.append(
            {"unknown_text": texts[i], "author": str(predictions[i]), "score": decision_function[0][i]})
    if output_directory is None:
        output_directory = dirname(dirname(abspath(__file__))) + "/AAS_Results/LSVM"
    if not isdir(output_directory):
        mkdir(output_directory)
    f = open(output_directory + '/answers.json', "w")
    dump({"answers": answers}, f, indent=2)
    f.close()
    return output_directory


def keselj_attribution(input_directory=None, verbose=False, output_directory=None):
    if input_directory is None:
        input_directory = dirname(dirname(abspath(__file__))) + \
                          "/AAS_Input"
    if output_directory is None:
        output_directory = dirname(dirname(abspath(__file__))) + "/AAS_Results/Keselj"

    if not isdir(output_directory):
        try:
            mkdir(output_directory)
        except FileExistsError:
            pass
    arguments = ["python3",
                 dirname(dirname(abspath(__file__))) +
                 "/AAS/Keselj/keselj03.py",
                 "-i", input_directory, "-o", output_directory]
    call(arguments)
    return output_directory


def teahan_attribution(input_directory=None, verbose=False, output_directory=None):
    if input_directory is None:
        input_directory = dirname(dirname(abspath(__file__))) + \
                          "/AAS_Input"
        if output_directory is None:
            output_directory = dirname(dirname(abspath(__file__))) + "/AAS_Results/Teahan"

    if not isdir(output_directory):
        mkdir(output_directory)

    arguments = ["python3",
                 dirname(dirname(abspath(__file__))) +
                 "/AAS/Teahan/teahan03.py",
                 "-i" + input_directory, "-o" + output_directory]
    call(arguments)
    return output_directory


def koppel_attribution(input_directory=None, output_directory=None):
    if input_directory is None:
        input_directory = dirname(dirname(abspath(__file__))) + \
                          "/AAS_Input"
    if output_directory is None:
        output_directory = dirname(dirname(abspath(__file__))) + "/AAS_Results/Koppel"

    if not isdir(output_directory):
        mkdir(output_directory)

    arguments = ["python3",
                 dirname(dirname(abspath(__file__))) +
                 "/AAS/Koppel/koppel11.py",
                 "-i" + input_directory, "-o" + output_directory]
    call(arguments)
    return output_directory


def benedetto_attribution(input_directory=None):
    if input_directory is None:
        input_directory = dirname(dirname(abspath(__file__))) + \
                          "/AAS_Input"
    output_directory = dirname(dirname(abspath(__file__))) + \
                       "/AAS_Results/Benedetto"

    if not isdir(output_directory):
        mkdir(output_directory)

    arguments = ["python3",
                 dirname(dirname(abspath(__file__))) +
                 "/AAS/Benedetto/benedetto02.py",
                 "-i" + input_directory, "-o" + output_directory]
    call(arguments)


def jairescalante_attribution(input_directory=None, verbose=False):
    if input_directory is None:
        input_directory = dirname(dirname(abspath(__file__))) + \
                          "/AAS_Input"
    output_directory = dirname(dirname(abspath(__file__))) + \
                       "/AAS_Results/Jairescalante"

    if not isdir(output_directory):
        mkdir(output_directory)

    arguments = ["java",
                 dirname(dirname(abspath(__file__))) +
                 "/AAS/Jairescalante/src/jairescalante11/escalante.jar",
                 "-i" + input_directory, "-o" + output_directory + '/answers.json']
    call(arguments)


def stamatatos_attribution(input_directory=None, verbose=False):
    if input_directory is None:
        input_directory = dirname(dirname(abspath(__file__))) + \
                          "/AAS_Input"
    output_directory = dirname(dirname(abspath(__file__))) + \
                       "/AAS_Results/Stamatatos"

    if not isdir(output_directory):
        mkdir(output_directory)

    arguments = ["python3",
                 dirname(dirname(abspath(__file__))) +
                 "/AAS/Stamatatos/stamatatos07.py",
                 "-i" + input_directory, "-o" + output_directory]
    call(arguments)


def muttenthaler_attribution(input_directory=None, verbose=False):
    if input_directory is None:
        input_directory = dirname(dirname(abspath(__file__))) + \
                          "/AAS_Input"
    output_directory = dirname(dirname(abspath(__file__))) + \
                       "/AAS_Results/Muttenthaler"

    if not isdir(output_directory):
        mkdir(output_directory)

    from Code.Muttenthaler import baseline
    baseline(input_directory, output_directory, word_range=(1,3), dist_range=(1,3), char_range=(2,5), use_LSA=True)


def schaetti_attribution(input_directory=None, verbose=False):
    if input_directory is None:
        input_directory = dirname(dirname(abspath(__file__))) + \
                          "/AAS_Input"
    output_directory = dirname(dirname(abspath(__file__))) + \
                       "/AAS_Results/Schaetti"

    if not isdir(output_directory):
        mkdir(output_directory)

    arguments = ["python3",
                 dirname(dirname(abspath(__file__))) +
                 "/AAS/Schaetti/main.py",
                 "--input-dataset", input_directory, "--output-dir", output_directory]
    call(arguments)


def gagala_attribution(input_directory=None, verbose=False):
    if input_directory is None:
        input_directory = dirname(dirname(abspath(__file__))) + \
                          "/AAS_Input"
    output_directory = dirname(dirname(abspath(__file__))) + \
                       "/AAS_Results/Gagala"

    if not isdir(output_directory):
        mkdir(output_directory)

    arguments = ["python3",
                 dirname(dirname(abspath(__file__))) +
                 "/AAS/Gagala/Run.py",
                 "-c", input_directory, "-o", output_directory]
    call(arguments)

def sidorov_attribution(input_directory=None, verbose=False):
    if input_directory is None:
        input_directory = dirname(dirname(abspath(__file__))) + \
                          "/AAS_Input"
    output_directory = dirname(dirname(abspath(__file__))) + \
                       "/AAS_Results/Sidorov"

    if not isdir(output_directory):
        mkdir(output_directory)

    arguments = ['java',
                 dirname(dirname(abspath(__file__))) +
                 "/AAS/Sidorov/src/main/Attributor",
                 "-i" + input_directory, "-o" + output_directory]
    call(arguments)


def seroussi_attribution(input_directory=None, verbose=False):
    if input_directory is None:
        input_directory = dirname(dirname(abspath(__file__))) + \
                          "/AAS_Input"
    output_directory = dirname(dirname(abspath(__file__))) + \
                       "/AAS_Results/Seroussi"

    if not isdir(output_directory):
        mkdir(output_directory)

    arguments = ["python3",
                 dirname(dirname(abspath(__file__))) +
                 "/AAS/Seroussi/main.py",
                 input_directory, output_directory]
    call(arguments)


def arun_attribution(input_directory=None, verbose=False):
    if input_directory is None:
        input_directory = dirname(dirname(abspath(__file__))) + \
                          "/AAS_Input"
    output_directory = dirname(dirname(abspath(__file__))) + \
                       "/AAS_Results/Arun"

    if not isdir(output_directory):
        mkdir(output_directory)

    arguments = ["python3",
                 dirname(dirname(abspath(__file__))) +
                 "/AAS/Arun/arun09.py",
                 '-i' + input_directory, '-o' + output_directory]
    call(arguments)


def burrows_attribution(input_directory=None, verbose=False):
    if input_directory is None:
        input_directory = dirname(dirname(abspath(__file__))) + \
                          "/AAS_Input"
    output_directory = dirname(dirname(abspath(__file__))) + \
                       "/AAS_Results/Burrows"

    if not isdir(output_directory):
        mkdir(output_directory)

    arguments = ["python3",
                 dirname(dirname(abspath(__file__))) +
                 "/AAS/Burrows/burrows02.py",
                 '-i' + input_directory, '-o' + output_directory]
    call(arguments)


def devel_attribution(input_directory=None, verbose=False):
    if input_directory is None:
        input_directory = dirname(dirname(abspath(__file__))) + \
                          "/AAS_Input"
    output_directory = dirname(dirname(abspath(__file__))) + \
                       "/AAS_Results/Devel"

    if not isdir(output_directory):
        mkdir(output_directory)

    arguments = [dirname(dirname(abspath(__file__))) +
                 "/AAS/Devel/devel.exe",
                 '-i' + input_directory, '-o' + output_directory]
    call(arguments)


def garciacumbreras_attribution(input_directory=None, verbose=False):
    if input_directory is None:
        input_directory = dirname(dirname(abspath(__file__))) + \
                          "/AAS_Input"
    output_directory = dirname(dirname(abspath(__file__))) + \
                       "/AAS_Results/Garciacumbreras"

    if not isdir(output_directory):
        mkdir(output_directory)

    arguments = ["python3",
                 dirname(dirname(abspath(__file__))) +
                 "/AAS/Garciacumbreras/classify_comp.py",
                 "-i" + input_directory, "-o" + output_directory]
    call(arguments)


def khmelev_attribution(input_directory=None, verbose=False):
    if input_directory is None:
        input_directory = dirname(dirname(abspath(__file__))) + \
                          "/AAS_Input"
    output_directory = dirname(dirname(abspath(__file__))) + \
                       "/AAS_Results/Khmelev"

    if not isdir(output_directory):
        mkdir(output_directory)

    arguments = [dirname(dirname(abspath(__file__))) +
                 "/AAS/Khmelev/./khmelev03",
                 "-i" + input_directory, "-o" + output_directory]
    call(arguments)


def glad_verification(input_directory=None, output_directory=None):
    if output_directory is None:
        output_directory = "/CASIS Versions/AVS/CASIS-25_Dataset - Mihaylova - Verification"
    if input_directory is None:
        input_directory = dirname(dirname(abspath(__file__))) + '/AVS_Input'
    model = dirname(dirname(abspath(__file__))) + \
            "/AVS/GLAD/model"

    if isdir(output_directory):
        rmtree(output_directory)

    mkdir(output_directory)
    arguments = ["python3",
                 dirname(dirname(abspath(__file__))) +
                 '/AVS/GLAD/glad-main.py',
                 '--test', input_directory, '--m', model, '--o',
                 dirname(dirname(abspath(__file__))) + output_directory]
    arguments = ["python3",
                 dirname(dirname(abspath(__file__))) +
                 '/AVS/GLAD/glad-main.py',
                 '--test', input_directory, '--model', model, '--out', output_directory]

    call(arguments)


def caravel_verification(input_directory, output_directory):
    testing = dirname(dirname(abspath(__file__))) + \
              "/CASIS Versions/CASIS-25_Dataset - Input"

    arguments = ["python2",
                 dirname(dirname(abspath(__file__))) +
                 '/AVS/Caravel/pan-ensemble',
                 '-i', input_directory, '-o', output_directory]
    call(arguments)


def authorid_pfp_verification_train(input_directory):
    model = dirname(dirname(abspath(__file__))) + \
            "/AVS/AuthorID_PFP/modelDir"

    arguments = ["python2",
                 dirname(dirname(abspath(__file__))) +
                 '/AVS/AuthorID_PFP/train.py',
                 '-i', input_directory, '-o', model]

    call(arguments)


def authorid_pfp_verification_test(input_directory, output_directory):
    model = dirname(dirname(abspath(__file__))) + \
            "/AVS/AuthorID_PFP/modelDir"

    arguments = ["python2",
                 dirname(dirname(abspath(__file__))) +
                 '/AVS/AuthorID_PFP/test.py',
                 '-i', input_directory, '-m', model, '-o', output_directory]

    call(arguments)


def run_author_masking(dataset_directory, known_directory="", masking_used=None, verbose=False,
                       output_directory=None, remove_directory=True):

    if masking_used is None:
        masking_used = ['Allred', 'Castro', 'Mihaylova', 'Rahgouy']

    for index in range(len(masking_used)):
        masking_used[index] = masking_used[index].lower()

    if 'allred' in masking_used:
        if verbose:
            print("Running Allred Masking...")
        start = time()

        allred_masking(dataset_directory, output_directory=output_directory, remove_directory=remove_directory)
        finish = time()

    if 'castro' in masking_used:
        if verbose:
            print("Running Castro Masking...")
        start = time()
        castro_masking(dataset_directory, output_directory=output_directory, remove_directory=remove_directory)
        finish = time()

    if 'mihaylova' in masking_used:
        if verbose:
            print("Running Mihaylova Masking...")
        start = time()
        results_directory = mihaylova_masking(dataset_directory, output_directory=output_directory, remove_directory=remove_directory)
        get_obfuscation_mihaylova(results_directory)
        finish = time()

    if 'rahgouy' in masking_used:
        if verbose:
            print("Running Rahgouy Masking...")
        start = time()
        rahgouy_preprocessing(dataset_directory, known_directory)
        rahgouy_masking(dataset_directory, known_directory, output_directory=output_directory, remove_directory=remove_directory)
        finish = time()
        get_obfuscation_rahgouy(output_directory)

    if 'kocher' in masking_used:
        if verbose:
            print("Running Kocher Masking...")
        start = time()
        if output_directory is None:
            author_masking = dirname(dirname(abspath(__file__))) + '/AMT_Results'
            output_directory = author_masking + '/Kocher/'
        kocher_masking(known_directory=dataset_directory, unknown_directory=known_directory, output_directory=output_directory)
        get_obfuscation_mihaylova(output_directory)
        finish = time()

def run_author_attribution(attribution_used, dataset_directory=None, preprocess=True, verbose=False,
                           validation=False, train_files=None, validation_files=None, unknown_files=None,
                           known_files=None, validation_directory=None, unknown_directory=None,
                           known_directory=None, input_directory=None, truth_file=None, output_directory=None, runs=5):
    answers = {}
    cdaa_preprocess = preprocess

    unknown_mapping = None

    for index in range(len(attribution_used)):
        attribution_used[index] = attribution_used[index].lower()
    if 'lsvm' in attribution_used:
        if preprocess:
            train_files, validation_files, unknown_files, unknown_mapping = author_attribution_preprocessing(
                verbose=verbose, train_directory=known_directory, unknown_directory=unknown_directory)
        if verbose:
            print('Running LSVM Attribution...')
        output_directory = lsvm_attribution(input_directory=input_directory)
        lsvm_answers = get_answers('LSVM', answers_file=output_directory + '/answers.json')
        lsvm_accuracy = get_answers_accuracy('LSVM', truth_file=truth_file, answers_file=output_directory + '/answers.json')
        if verbose:
            print('LSVM accuracy: ' + str(lsvm_accuracy) + '%')
            print()
        answers.update({'LSVM': lsvm_answers})

    if 'keselj' in attribution_used:
        if preprocess:
            train_files, validation_files, unknown_files, unknown_mapping = author_attribution_preprocessing(
                dataset_directory=dataset_directory, verbose=verbose, train_directory=known_directory,
                validation_directory=validation_directory, unknown_directory=unknown_directory, validation=True)
            preprocess = not validation
        if verbose:
            print('Running Keselj Attribution...')

        potential_answers = {}
        keselj_answers = []
        keselj_scores = {}
        keselj_accuracy = 0
        for run in range(1):
            output_directory = keselj_attribution(input_directory=input_directory, output_directory=output_directory)
            run_answers = get_answers('Keselj', answers_file=output_directory + '/answers.json')
            for answer in run_answers:
                if answer['unknown_text'] not in potential_answers:
                    potential_answers[answer['unknown_text']] = []
                if answer['unknown_text'] not in keselj_scores:
                    keselj_scores[answer['unknown_text']] = []
                potential_answers[answer['unknown_text']].append(answer['author'])
                keselj_scores[answer['unknown_text']].append(answer['score'])
            keselj_accuracy = (keselj_accuracy * run + get_answers_accuracy('Keselj', truth_file=truth_file, answers_file=output_directory + '/answers.json')) / (run + 1)
        for answer in potential_answers:
            selected_author = Counter(potential_answers[answer]).most_common(1)[0][0]
            author_score = mean(keselj_scores[answer])
            keselj_answers.append({'unknown_text': answer, 'author': selected_author, 'score': author_score})
        if verbose:
            print('Keselj accuracy: ' + str(keselj_accuracy) + '%')
            print()
        answers.update({'Keselj': keselj_answers})

    if 'teahan' in attribution_used:
        if preprocess:
            train_files, validation_files, unknown_files, unknown_mapping = author_attribution_preprocessing(
                dataset_directory=dataset_directory, verbose=verbose, train_directory=known_directory,
                validation_directory=validation_directory, unknown_directory=unknown_directory)
        if verbose:
            print('Running Teahan Attribution...')
        output_directory = teahan_attribution(input_directory=input_directory, verbose=verbose, output_directory=output_directory)
        teahan_answers = get_answers('Teahan', answers_file=output_directory + '/answers.json')
        teahan_accuracy = get_answers_accuracy('Teahan', truth_file=truth_file, answers_file=output_directory + '/answers.json')
        if verbose:
            print('Teahan accuracy: ' + str(teahan_accuracy) + '%')
            print()
        answers.update({'Teahan': teahan_answers})

    if 'koppel' in attribution_used:
        if preprocess:
            train_files, validation_files, unknown_files, unknown_mapping = author_attribution_preprocessing(
                dataset_directory=dataset_directory, verbose=verbose, train_directory=known_directory,
                validation_directory=validation_directory, unknown_directory=unknown_directory)
            preprocess = False
        if verbose:
            print('Running Koppel Attribution...')
        potential_answers = {}
        koppel_answers = []
        koppel_scores = {}
        koppel_accuracy = 0
        for run in range(runs):
            output_directory = koppel_attribution(input_directory=input_directory, output_directory=output_directory)
            run_answers = get_answers('Koppel', answers_file=output_directory + '/answers.json')
            for answer in run_answers:
                if answer['unknown_text'] not in potential_answers:
                    potential_answers[answer['unknown_text']] = []
                if answer['unknown_text'] not in koppel_scores:
                    koppel_scores[answer['unknown_text']] = []
                potential_answers[answer['unknown_text']].append(answer['author'])
                koppel_scores[answer['unknown_text']].append(answer['score'])
        for answer in potential_answers:
            selected_author = Counter(potential_answers[answer]).most_common(1)[0][0]
            author_score = mean(koppel_scores[answer])
            koppel_answers.append({'unknown_text': answer, 'author': selected_author, 'score': author_score})
            with open(output_directory + '/answers.json', 'w+') as answers_json:
                dump(koppel_answers, answers_json)
            koppel_accuracy = get_answers_accuracy('Koppel', truth_file=truth_file, answers_file=output_directory + '/answers.json')
        if verbose:
            print('Koppel accuracy: ' + str(koppel_accuracy) + '%\n')
            print()
        answers.update({'Koppel': koppel_answers})

    if 'benedetto' in attribution_used:
        if preprocess:
            train_files, validation_files, unknown_files, unknown_mapping = author_attribution_preprocessing(
                dataset_directory=dataset_directory, verbose=verbose, train_directory=known_directory,
                validation_directory=validation_directory, unknown_directory=unknown_directory)
            preprocess = False
        if verbose:
            print('Running Benedetto Attribution...')
        benedetto_attribution()
        benedetto_answers = get_answers('Benedetto')
        benedetto_accuracy = get_answers_accuracy('Benedetto')
        if verbose:
            print('Benedetto accuracy: ' + str(benedetto_accuracy) + '%')
            print()
        answers.update({'Benedetto': benedetto_answers})

    if 'stamatatos'in attribution_used:
        if preprocess:
            train_files, validation_files, unknown_files, unknown_mapping = author_attribution_preprocessing(
                dataset_directory=dataset_directory, verbose=verbose, train_directory=known_directory,
                validation_directory=validation_directory, unknown_directory=unknown_directory)
            preprocess = False
        print('Running Stamatatos Attribution...')
        stamatatos_attribution()
        stamatatos_answers = get_answers('Stamatatos')
        stamatatos_accuracy = get_answers_accuracy('Stamatatos')
        print('Stamatatos accuracy: ' + str(stamatatos_accuracy) + '%')
        print()
        answers.update({'Stamatatos': stamatatos_answers})

    if 'jairescalante' in attribution_used:
        if preprocess:
            train_files, validation_files, unknown_files, unknown_mapping = author_attribution_preprocessing(
                dataset_directory=dataset_directory, verbose=verbose, train_directory=known_directory,
                validation_directory=validation_directory, unknown_directory=unknown_directory)
            preprocess = False
        print('Running Jairescalante Attribution...')
        jairescalante_attribution()
        jairescalante_answers = get_answers('Jairescalante')
        jairescalante_accuracy = get_answers_accuracy('Jairescalante')
        print('Jairescalante accuracy: ' + str(jairescalante_accuracy) + '%')
        print()
        answers.update({'Jairescalante': jairescalante_answers})

    if 'burrows' in attribution_used:
        if preprocess:
            train_files, validation_files, unknown_files, unknown_mapping = author_attribution_preprocessing(
                dataset_directory=dataset_directory, verbose=verbose, train_directory=known_directory,
                validation_directory=validation_directory, unknown_directory=unknown_directory)
            preprocess = False
        print('Running Burrows Attribution...')
        burrows_attribution()
        burrows_answers = get_answers('Burrows')
        burrows_accuracy = get_answers_accuracy('Burrows')
        print('Burrows accuracy: ' + str(burrows_accuracy) + '%')
        print()
        answers.update({'Burrows': burrows_answers})

    if 'arun' in attribution_used:
        if cdaa_preprocess:
            train_files, validation_files, unknown_files, unknown_mapping = author_attribution_preprocessing(
                dataset_directory=dataset_directory, verbose=verbose, train_directory=known_directory,
                validation_directory=validation_directory, unknown_directory=unknown_directory)
            cdaa_preprocess = False
        print('Running Arun Attribution...')
        arun_attribution()
        arun_answers = get_answers('Arun')
        arun_accuracy = get_answers_accuracy('Arun')
        print('Arun accuracy: ' + str(arun_accuracy) + '%')
        print()
        answers.update({'Arun': arun_answers})

    if 'muttenthaler' in attribution_used:
        if cdaa_preprocess:
            train_files, unknown_files = cdaa_preprocessing(dataset_directory=dataset_directory, verbose=verbose,
                                                            train_directory=known_directory,
                                                            unknown_directory=unknown_directory)
            cdaa_preprocess = False
        print('Running Muttenthaler Attribution...')
        muttenthaler_attribution()
        muttenthaler_answers = get_answers('Muttenthaler')
        muttenthaler_accuracy = get_answers_accuracy('Muttenthaler')
        print('Muttenthaler accuracy: ' + str(muttenthaler_accuracy) + '%')
        print()
        answers.update({'Muttenthaler': muttenthaler_answers})

    if 'schaetti' in attribution_used:
        if cdaa_preprocess:
            train_files, unknown_files = cdaa_preprocessing(dataset_directory=dataset_directory, verbose=verbose,
                                                            train_directory=known_directory,
                                                            unknown_directory=unknown_directory)
            cdaa_preprocess = False
        print('Running Schaetti Attribution...')
        schaetti_attribution()
        schaetti_answers = get_answers('Schaetti')
        schaetti_accuracy = get_answers_accuracy('Schaetti')
        print('Schaetti accuracy: ' + str(schaetti_accuracy) + '%')
        print()
        answers.update({'Schaetti': schaetti_answers})

    if 'garciacumbreras' in attribution_used:
        if cdaa_preprocess:
            train_files, unknown_files = cdaa_preprocessing(dataset_directory=dataset_directory, verbose=verbose,
                                                            train_directory=known_directory,
                                                            unknown_directory=unknown_directory)
            cdaa_preprocess = False
        print('Running Garciacumbreras Attribution...')
        garciacumbreras_attribution()
        garciacumbreras_answers = get_answers('Garciacumbreras')
        garciacumbreras_accuracy = get_answers_accuracy('Garciacumbreras')
        print('Garciacumbreras accuracy: ' + str(garciacumbreras_accuracy) + '%')
        print()
        answers.update({'Garciacumbreras': garciacumbreras_answers})

    if 'gagala' in attribution_used:
        print(dataset_directory)
        if cdaa_preprocess:
            train_files, unknown_files = cdaa_preprocessing(dataset_directory=dataset_directory, verbose=verbose,
                                                            train_directory=known_directory,
                                                            unknown_directory=unknown_directory)
            cdaa_preprocess = False
        print('Running Gagala Attribution...')
        gagala_attribution()
        gagala_answers = get_answers('Gagala')
        gagala_accuracy = get_answers_accuracy('Gagala')
        print('Gagala accuracy: ' + str(gagala_accuracy) + '%')
        print()
        answers.update({'Gagala': gagala_answers})

    if 'khmelev' in attribution_used:
        if preprocess:
            train_files, validation_files, unknown_files, unknown_mapping = author_attribution_preprocessing(
                dataset_directory=dataset_directory, verbose=verbose, train_directory=known_directory,
                validation_directory=validation_directory, unknown_directory=unknown_directory)
            preprocess = False
        print('Running Khmelev Attribution...')
        khmelev_attribution()
        khmelev_answers = get_answers('Khmelev')
        khmelev_accuracy = get_answers_accuracy('Khmelev')
        print('Khmelev accuracy: ' + str(khmelev_accuracy) + '%')
        print()
        answers.update({'Gagala': khmelev_answers})


    # fitness = fitness_function(answers=answers, truth_file=truth_file)

    return answers, None, unknown_mapping


def get_beam_results(data_directory, att_list):
    if not isdir(data_directory):
        data_directory = dirname(dirname(abspath(__file__))) + '/' + data_directory
    for directory in listdir(data_directory):
        if not isdir(data_directory + '/' + directory):
            continue
        print(data_directory + '/' + directory)
        if len(listdir(data_directory + '/' + directory)) > 1:
            for file in listdir(data_directory + '/' + directory):
                if isdir(data_directory + '/' + directory + '/' + file):
                    if isfile(data_directory + '/' + directory + '/' + file + '/original.txt'):
                        copy(data_directory + '/' + directory + '/' + file + '/original.txt',
                             data_directory + '/' + directory + '/' + file + '.txt')
                        print(data_directory + '/' + directory + '/' + file + '/original.txt')
                        print(data_directory + '/' + directory + '/' + file + '.txt')
                        rmtree(data_directory + '/' + directory + '/' + file)
                    else:
                        raise Exception("The following file is a folder: " + file)

            answers, fitness, unknown_mapping = run_author_attribution(attribution_used=att_list,
                                                            unknown_directory=data_directory + '/' + directory,
                                                            known_directory=dirname(dirname(abspath(__file__))) + '/CASIS Versions/CASIS-25_Dataset - Known',
                                                            runs=3, preprocess=True)

            minimum_fitness = min(fitness.values())
            remove_file = unknown_mapping[list(fitness.keys())[list(fitness.values()).index(minimum_fitness)]]
            copy(data_directory + '/' + directory + '/' + remove_file, data_directory + '/' + remove_file)
            rmtree(data_directory + '/' + directory)
        else:
            copy(data_directory + '/' + directory + '/' + directory + '.txt', data_directory + '/' + directory + '.txt')
            rmtree(data_directory + '/' + directory)
