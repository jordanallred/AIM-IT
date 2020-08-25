from os import listdir
from os import remove
from os.path import isfile, dirname, abspath

from matplotlib import pyplot as plt
from numpy import array
from numpy import concatenate
from numpy import max, min, arange
from numpy.random import randint
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_selection import RFE
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
from sklearn.svm import LinearSVC

directory = dirname(dirname(abspath(__file__))) + "/CASIS Versions/CASIS-25_Dataset/"
input_file = dirname(dirname(abspath(__file__))) + "/Code/Allred/input.txt"
original_file = dirname(dirname(abspath(__file__))) + "/Code/Allred/original.txt"
output_file = dirname(dirname(abspath(__file__))) + "/Code/Allred/output.txt"
error = dirname(dirname(abspath(__file__))) + "/Code/Allred/error.txt"
best_file = dirname(dirname(abspath(__file__))) + "/Code/Allred/best_result.txt"
language = dirname(dirname(abspath(__file__))) + "/Code/Allred/language.txt"
spanish_output = dirname(dirname(abspath(__file__))) + "/Code/Allred/spanish.txt"
catalan_output = dirname(dirname(abspath(__file__))) + "/Code/Allred/catalan.txt"
gallegan_output = dirname(dirname(abspath(__file__))) + "/Code/Allred/gallegan.txt"
wordnet_output = dirname(dirname(abspath(__file__))) + "/Code/Allred/wordnet.txt"
pan_file = dirname(dirname(abspath(__file__))) + "/Code/Allred/PAN.txt"
target_author = 0

corpus_vectorizer: CountVectorizer

def files_to_corpus(directory_name: str):
    file_list = listdir(directory_name)
    corpus = []

    for file in file_list:
        path = directory_name + file
        with open(path, errors='ignore') as open_file:
            file_data = open_file.read()
        corpus.append(file_data)

    return corpus


def file_to_corpus(file_name: str):
    corpus = []

    with open(file_name, encoding='utf8') as open_file:
        file_data = open_file.read()
    corpus.append(file_data)

    return corpus


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
    #global stop_words
    #vectorizer = CountVectorizer(stop_words=stop_words)
    vectorizer = CountVectorizer()
    model = vectorizer.fit_transform(corpus)
    global corpus_vectorizer
    corpus_vectorizer = vectorizer
    vocabulary = model.toarray()
    vocabulary = normalize(vocabulary)

    return vocabulary


def feature_vocabulary(feature: list):
    global corpus_vocabulary, stop_words
    #vectorizer = CountVectorizer(vocabulary=corpus_vectorizer.vocabulary_, stop_words=stop_words)
    vectorizer = CountVectorizer(vocabulary=corpus_vectorizer.vocabulary_)
    model = vectorizer.fit_transform(feature)
    vocabulary = model.toarray()
    vocabulary = normalize(vocabulary)

    return vocabulary


def get_authors_casis():
    global directory

    author_list = []
    if directory.__contains__("CASIS"):
        for index in range(100):
            author_list.append(array(index // 4))

    else:
        author_directory = listdir(directory)
        for author in author_directory:
            author_title = author.split('_')
            author_name = author_title[1]
            author_list.append(author_name[:-4])
    return array(author_list)


def extract_features(directory_name: str):
    unigram_vector, bigram_vector = [], []
    file_list = listdir(directory_name)

    authors_list = get_authors_casis()

    for file in file_list:
        if file.__contains__('.txt'):
            path = directory_name + file
            unigram_vector.append(character_unigram(path))
            bigram_vector.append(character_bigram(path))


    casis_corpus = files_to_corpus(directory)
    vocabulary_vector = bag_of_words(casis_corpus)

    feature_vector_3 = [0] * len(unigram_vector)
    feature_vector_7 = [0] * len(unigram_vector)

    for index in range(len(feature_vector_7)):
        feature_vector_3[index] = concatenate((unigram_vector[index], bigram_vector[index], vocabulary_vector[index]))
        feature_vector_7[index] = vocabulary_vector[index]

    return array(feature_vector_3), array(feature_vector_7), array(authors_list)
    # return array(feature_vector_7), array(authors_list)


def extract_features_single(file_name: str):
    unigram_vector, bigram_vector = [], []

    unigram_vector.append(character_unigram(file_name))
    bigram_vector.append(character_bigram(file_name))
    data = file_to_corpus(file_name)
    vocabulary_vector = feature_vocabulary(data)

    unigram_vector = array(unigram_vector)
    unigram_vector = unigram_vector.reshape(1,95)

    feature_vector_3 = [0] * len(unigram_vector)
    feature_vector_7 = [0] * len(unigram_vector)

    for index in range(len(feature_vector_7)):
        feature_vector_3[index] = concatenate((unigram_vector[index], bigram_vector[index], vocabulary_vector[index]))
        feature_vector_7[index] = vocabulary_vector[index]

    return array(feature_vector_3), array(feature_vector_7)


def lsvm_classifier(authors: array, features: array, feature_max = 1000):
    train_labels, test_labels, train_data, test_data = train_test_split(authors, features, test_size=0.10)
    model = LinearSVC()
    selector = RFE(model, feature_max, 50, verbose=0)
    selector = selector.fit(train_data, train_labels)
    predictions = selector.predict(test_data)
    #for feature in range(len(features)):
        #for index in range(len(features[feature])):
            #features[feature][index] *= feature_mask[index]
    model.fit(train_data, train_labels)
    predictions = model.predict(test_data)
    accuracy = accuracy_score(predictions, test_labels)

    return accuracy


def lsvm_classifier_adversary(authors: array, features: array, test_feature: array, verbose: int, target: int):
    train_labels, train_data = authors, features
    model = LinearSVC()
    model.fit(train_data, train_labels)
    prediction = model.predict(test_feature)
    decision_function = model.decision_function(test_feature)

    green_bar_x, green_bar_y, red_bar_x, red_bar_y, blue_bar_x, blue_bar_y = [], [], [], [], [], []

    max_element = max(decision_function)
    min_element = min(decision_function)

    max_value = 1
    min_value = 0

    for decision in range(len(decision_function[0])):
        decision_function[0][decision] = (decision_function[0][decision] - min_element) / (max_element - min_element)
        decision_function[0][decision] = decision_function[0][decision] * (max_value - min_value) + min_value

    for decision in range(len(decision_function[0])):
        if decision_function[0][decision] == max(decision_function):
            green_bar_x.append(decision)
            green_bar_y.append(decision_function[0][decision])
        else:
            if decision == target:
                blue_bar_x.append(decision)
                blue_bar_y.append(decision_function[0][decision])
            else:
                red_bar_x.append(decision)
                red_bar_y.append(decision_function[0][decision])
    if verbose == 1:
        plt.bar(red_bar_x, red_bar_y, color='red')
        plt.bar(blue_bar_x, blue_bar_y, color = 'blue')
        plt.bar(green_bar_x, green_bar_y, color='green')
        plt.xticks(arange(0, len(decision_function[0]), step=1))
        plt.xlabel("Author")
        plt.ylabel("Decision Score")
        plt.ion()
        plt.pause(2)
        plt.show()
        plt.pause(2)
        plt.cla()

    #'''
    rmse = 0
    maximum = 0
    for decision in range(len(decision_function[0])):
        if decision == target:
            rmse += decision_function[0][decision]
        elif maximum < decision_function[0][decision]:
            maximum = decision_function[0][decision]
    rmse -= maximum

    #print("Error: " + str(rmse) + "\n")

    '''
    rmse = 100
    for decision in range(len(decision_function)):
        if decision != target:
            if decision_function[0][target] - decision_function[0][decision] < rmse:
                rmse = decision_function[0][target] - decision_function[0][decision]
    #'''

    if isfile(error):
        with open(error, 'r') as read_file:
            content = read_file.read()
        score = content.split(',')
        if float(score[1]) > rmse:
            remove(error)
            with open(error, 'w+') as write_file:
                with open(language, 'r') as read_file:
                    write_file.write(read_file.read() + "," + str(rmse))
    else:
        with open(error, 'w+') as write_file:
            with open(language, 'r') as read_file:
                write_file.write(read_file.read() + "," + str(rmse))


def lsvm_classifier_original(authors: array, features: array, test_feature: array, verbose: int):
    global target_author
    train_labels, train_data = authors, features
    model = LinearSVC()
    model.fit(train_data, train_labels)
    prediction = model.predict(test_feature)
    decision_function = model.decision_function(test_feature)

    green_bar_x, green_bar_y, red_bar_x, red_bar_y, blue_bar_x, blue_bar_y = [], [], [], [], [], []

    max_element = max(decision_function)
    min_element = min(decision_function)

    max_value = 1
    min_value = 0

    for decision in range(len(decision_function[0])):
        decision_function[0][decision] = (decision_function[0][decision] - min_element) / (max_element - min_element)
        decision_function[0][decision] = decision_function[0][decision] * (max_value - min_value) + min_value

    for decision in range(len(decision_function[0])):
        if decision_function[0][decision] == max(decision_function):
            green_bar_x.append(decision)
            green_bar_y.append(decision_function[0][decision])
        else:
            if decision == target_author:
                blue_bar_x.append(decision)
                blue_bar_y.append(decision_function[0][decision])
            else:
                red_bar_x.append(decision)
                red_bar_y.append(decision_function[0][decision])
    if verbose == 1:
        plt.bar(red_bar_x, red_bar_y, color='red')
        plt.bar(blue_bar_x, blue_bar_y, color = 'blue')
        plt.bar(green_bar_x, green_bar_y, color='green')
        plt.xticks(arange(0, len(decision_function[0]), step=1))
        plt.xlabel("Author")
        plt.ylabel("Decision Score")
        plt.ion()
        plt.pause(1)
        plt.show()
        plt.pause(1)
        plt.cla()


def feature_masks(authors: array, features: array, num_features: int, population_size: int, iterations: int, mutation_rate = 0.05, generation_gap = 2/5):
    population = []
    feature_mask = []
    for member in range(population_size):
        for index in range(num_features):
            feature_mask.append(randint(0, 2))
        population.append(feature_mask)

    for iteration in range(iterations):
        population_accuracy = []
        for member in population:
            features_copy = features
            for feature in range(len(features)):
                for index in range(len(features_copy[feature])):
                    features_copy[feature][index] *= member[index]
                    if (randint(0, 100) / 100) < mutation_rate:
                        features_copy[feature][index] = 1 - features_copy[feature][index]
            population_accuracy.append(100 * lsvm_classifier(authors, features_copy))
        population_accuracy_copy = population_accuracy.copy()
        population_accuracy_copy.sort(reverse=True)
        cutoff = population_accuracy_copy[len(population) - 1 - round(population_size * generation_gap)]
        index, deleted = 0, 0
        while index < len(population) and deleted < round(population_size * generation_gap):
            if population_accuracy[index] > 80:
                print(population[index])
            if population_accuracy[index] < cutoff:
                del population_accuracy[index]
                del population[index]
                deleted += 1
            index += 1
        while len(population) < population_size - 1:
            feature_mask = []
            for index in range(num_features):
                feature_mask.append(population[randint(0, len(population) // 3)][index])
                if (randint(0, 100) / 100) < mutation_rate:
                    feature_mask[index] = 1 - feature_mask[index]
            population.append(feature_mask)
        sum_acc = 0
        for index in range(len(population_accuracy_copy) // 5):
            sum_acc += population_accuracy_copy[index]

        print("iteration: " + str(iteration))
        print("average accuracy: " + str(sum_acc / (len(population_accuracy_copy) // 5)))
    print(population_accuracy)
    print(population)


def run(input_type: str, verbose: int, target: int):
    global target_author
    target_author = target
    v3, v7, authors_list = extract_features(directory)

    if input_type == "original":
        plt.title("CASIS-25 Original Predictions")
        original_data, garbage1 = extract_features_single(original_file)
        lsvm_classifier_original(authors_list, v3, original_data, verbose)
    elif input_type == "wordnet":
        plt.title("CASIS-25 Wordnet Predictions")
        wordnet_data, garbage1 = extract_features_single(wordnet_output)
        lsvm_classifier_adversary(authors_list, v3, wordnet_data, verbose, target)
    elif input_type == "catalan":
        plt.title("CASIS-25 Catalan Predictions")
        catalan_data, garbage1 = extract_features_single(catalan_output)
        lsvm_classifier_adversary(authors_list, v3, catalan_data, verbose, target)
    elif input_type == "spanish":
        plt.title("CASIS-25 Spanish Predictions")
        spanish_data, garbage1 = extract_features_single(spanish_output)
        lsvm_classifier_adversary(authors_list, v3, spanish_data, verbose, target)
    elif input_type == "output":
        plt.title("CASIS-25 Output Predictions")
        output_data, garbage1 = extract_features_single(output_file)
        lsvm_classifier_original(authors_list, v3, output_data, verbose)
    elif input_type == "gallegan":
        plt.title("CASIS-25 Gallegan Predictions")
        gallegan_data, garbage1 = extract_features_single(gallegan_output)
        lsvm_classifier_adversary(authors_list, v3, gallegan_data, verbose, target)
    elif input_type == "best":
        plt.title("CASIS-25 Best Predictions")
        best_data, garbage1 = extract_features_single(best_file)
        lsvm_classifier_adversary(authors_list, v3, best_data, verbose, target)
    elif input_type == "pan":
        plt.title("CASIS-25 PAN Predictions")
        pan_data, garbage1 = extract_features_single(pan_file)
        lsvm_classifier_adversary(authors_list, v3, pan_data, verbose, target)
    else:
        print("Incorrect program execution")
