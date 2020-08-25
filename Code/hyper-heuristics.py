from dataclasses import dataclass
from graphviz import Digraph
from random import randint, random
from math import log, exp, sqrt
from os.path import dirname, abspath, isfile, isdir
from os import mkdir
from shutil import rmtree
import Code.jsonhandler as jsonhandler
from Code.Masking_Operations import *
from time import time
from collections import deque
from typing import List
from Code.Author_Masking import run_author_attribution
from numpy import std
from nltk.tokenize import word_tokenize
from shutil import move

def special_read(file_name):
    encodings = ['utf-8', 'windows-1250', 'windows-1252']
    for e in encodings:
        try:
            fh = open(file_name, 'r', encoding=e)
            fh.readlines()
            fh.seek(0)
        except UnicodeDecodeError:
            continue
        else:
            dfile = open(file_name, "r", encoding=e)
            s = dfile.read()
            dfile.close()
            return s
    raise Exception("No proper encoding was found for the file: " + file_name)


def Teahan(text):
    input_directory = dirname(dirname(abspath(__file__))) + "/AAS_Input"
    rmtree(input_directory + '/unknown')
    mkdir(input_directory + '/unknown')
    with open(input_directory + '/unknown/unknown01.txt', 'w+') as write_file:
        write_file.write(text)
    with open(input_directory + '/meta-file.json', 'w+') as json_file:
        json_file.write('{"folder": "unknown", "language": "EN", "encoding": "UTF8", "candidate-authors": '
                        '[{"author-name": "candidate1005"}, {"author-name": "candidate1007"}, '
                        '{"author-name": "candidate1021"}, {"author-name": "candidate1023"}, '
                        '{"author-name": "candidate1001"}, {"author-name": "candidate1003"}, '
                        '{"author-name": "candidate1018"}, {"author-name": "candidate1020"}, '
                        '{"author-name": "candidate1006"}, {"author-name": "candidate1022"}, '
                        '{"author-name": "candidate1004"}, {"author-name": "candidate1019"}, '
                        '{"author-name": "candidate1024"}, {"author-name": "candidate1000"}, '
                        '{"author-name": "candidate1002"}, {"author-name": "candidate1015"}, '
                        '{"author-name": "candidate1017"}, {"author-name": "candidate1008"}, '
                        '{"author-name": "candidate1011"}, {"author-name": "candidate1013"}, '
                        '{"author-name": "candidate1014"}, {"author-name": "candidate1016"}, '
                        '{"author-name": "candidate1012"}, {"author-name": "candidate1009"}, '
                        '{"author-name": "candidate1010"}], "unknown-texts": [{"unknown-text": "unknown01.txt"}]}')

    class Model(object):
        # cnt - count of characters read
        # modelOrder - order of the model
        # orders - List of Order-Objects
        # alphSize - size of the alphabet
        def __init__(self, order, alphSize):
            self.cnt = 0
            self.alphSize = alphSize
            self.modelOrder = order
            self.orders = []
            for i in range(order + 1):
                self.orders.append(Order(i))

        # print the model
        # TODO: Output becomes too long, reordering on the screen has to be made
        def printModel(self):
            s = "Total characters read: " + str(self.cnt) + "\n"
            for i in range(self.modelOrder + 1):
                self.printOrder(i)

        # print a specific order of the model
        # TODO: Output becomes too long, reordering on the screen has to be made
        def printOrder(self, n):
            o = self.orders[n]
            s = "Order " + str(n) + ": (" + str(o.cnt) + ")\n"
            for cont in o.contexts:
                if (n > 0):
                    s += "  '" + cont + "': (" + str(o.contexts[cont].cnt) + ")\n"
                for char in o.contexts[cont].chars:
                    s += "     '" + char + "': " + \
                         str(o.contexts[cont].chars[char]) + "\n"
            s += "\n"

        # updates the model with a character c in context cont
        def update(self, c, cont):
            if len(cont) > self.modelOrder:
                raise NameError("Context is longer than model order!")

            order = self.orders[len(cont)]
            if not order.hasContext(cont):
                order.addContext(cont)
            context = order.contexts[cont]
            if not context.hasChar(c):
                context.addChar(c)
            context.incCharCount(c)
            order.cnt += 1
            if (order.n > 0):
                self.update(c, cont[1:])
            else:
                self.cnt += 1

        # updates the model with a string
        def read(self, s):
            if (len(s) == 0):
                return
            for i in range(len(s)):
                cont = ""
                if (i != 0 and i - self.modelOrder <= 0):
                    cont = s[0:i]
                else:
                    cont = s[i - self.modelOrder:i]
                self.update(s[i], cont)

        # return the models probability of character c in content cont
        def p(self, c, cont):
            if len(cont) > self.modelOrder:
                raise NameError("Context is longer than order!")

            order = self.orders[len(cont)]
            if not order.hasContext(cont):
                if (order.n == 0):
                    return 1.0 / self.alphSize
                return self.p(c, cont[1:])

            context = order.contexts[cont]
            if not context.hasChar(c):
                if (order.n == 0):
                    return 1.0 / self.alphSize
                return self.p(c, cont[1:])
            return float(context.getCharCount(c)) / context.cnt

        # merge this model with another model m, esentially the values for every
        # character in every context are added
        def merge(self, m):
            if self.modelOrder != m.modelOrder:
                raise NameError("Models must have the same order to be merged")
            if self.alphSize != m.alphSize:
                raise NameError("Models must have the same alphabet to be merged")
            self.cnt += m.cnt
            for i in range(self.modelOrder + 1):
                self.orders[i].merge(m.orders[i])

        # make this model the negation of another model m, presuming that this
        # model was made my merging all models
        def negate(self, m):
            if self.modelOrder != m.modelOrder or self.alphSize != m.alphSize or self.cnt < m.cnt:
                raise NameError("Model does not contain the Model to be negated")
            self.cnt -= m.cnt
            for i in range(self.modelOrder + 1):
                self.orders[i].negate(m.orders[i])

    class Order(object):
        # n - whicht order
        # cnt - character count of this order
        # contexts - Dictionary of contexts in this order
        def __init__(self, n):
            self.n = n
            self.cnt = 0
            self.contexts = {}

        def hasContext(self, context):
            return context in self.contexts

        def addContext(self, context):
            self.contexts[context] = Context()

        def merge(self, o):
            self.cnt += o.cnt
            for c in o.contexts:
                if not self.hasContext(c):
                    self.contexts[c] = o.contexts[c]
                else:
                    self.contexts[c].merge(o.contexts[c])

        def negate(self, o):
            if self.cnt < o.cnt:
                raise NameError(
                    "Model1 does not contain the Model2 to be negated, Model1 might be corrupted!")
            self.cnt -= o.cnt
            for c in o.contexts:
                if not self.hasContext(c):
                    raise NameError(
                        "Model1 does not contain the Model2 to be negated, Model1 might be corrupted!")
                else:
                    self.contexts[c].negate(o.contexts[c])
            empty = [c for c in self.contexts if len(self.contexts[c].chars) == 0]
            for c in empty:
                del self.contexts[c]

    class Context(object):
        # chars - Dictionary containing character counts of the given context
        # cnt - character count of this context
        def __init__(self):
            self.chars = {}
            self.cnt = 0

        def hasChar(self, c):
            return c in self.chars

        def addChar(self, c):
            self.chars[c] = 0

        def incCharCount(self, c):
            self.cnt += 1
            self.chars[c] += 1

        def getCharCount(self, c):
            return self.chars[c]

        def merge(self, cont):
            self.cnt += cont.cnt
            for c in cont.chars:
                if not self.hasChar(c):
                    self.chars[c] = cont.chars[c]
                else:
                    self.chars[c] += cont.chars[c]

        def negate(self, cont):
            if self.cnt < cont.cnt:
                raise NameError(
                    "Model1 does not contain the Model2 to be negated, Model1 might be corrupted!")
            self.cnt -= cont.cnt
            for c in cont.chars:
                if (not self.hasChar(c)) or (self.chars[c] < cont.chars[c]):
                    raise NameError(
                        "Model1 does not contain the Model2 to be negated, Model1 might be corrupted!")
                else:
                    self.chars[c] -= cont.chars[c]
            empty = [c for c in self.chars if self.chars[c] == 0]
            for c in empty:
                del self.chars[c]

    # calculates the cross-entropy of the string 's' using model 'm'
    def h(m, s):
        n = len(s)
        h = 0
        for i in range(n):
            if i == 0:
                context = ""
            elif i <= m.modelOrder:
                context = s[0:i]
            else:
                context = s[i - m.modelOrder:i]
            h -= log(m.p(s[i], context), 2)
        return h / n

    # loads models of candidates in 'candidates' into 'models'

    # creates models of candidates in 'candidates'
    # updates each model with any files stored in the subdirectory of 'corpusdir' named with the candidates name
    # stores each model named under the candidates name in 'modeldir'
    def createModels():
        jsonhandler.loadTraining()
        for cand in candidates:
            models[cand] = Model(5, 256)
            for doc in jsonhandler.trainings[cand]:
                models[cand].read(jsonhandler.getTrainingText(cand, doc))

    # attributes the authorship, according to the cross-entropy ranking.
    # attribution is saved in json-formatted structure 'answers'
    def createAnswers():
        for doc in unknowns:
            hs = []
            for cand in candidates:
                hs.append(h(models[cand], jsonhandler.getUnknownText(doc)))
            maximum = max(hs)
            minimum = min(hs)
            author = candidates[hs.index(minimum)]
            for index in range(len(hs)):
                hs[index] = abs(hs[index] - maximum)
            hs.sort()
            score = hs[-1] / sum(hs)

            authors.append(author)
            scores.append(score)

    # commandline argument parsing, calling the necessary methods
    def main():
        corpusdir = input_directory
        jsonhandler.loadJson(corpusdir)

        createModels()
        createAnswers()
        return authors[0], scores[0]

    # initialization of global variables
    models = {}
    candidates = jsonhandler.candidates
    unknowns = jsonhandler.unknowns
    authors = []
    scores = []

    return main()


def LSVM(text):
    from numpy import array
    from numpy import concatenate
    from numpy import min
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.preprocessing import normalize
    from sklearn.svm import LinearSVC

    input_directory = dirname(dirname(abspath(__file__))) + "/AAS_Input"
    rmtree(input_directory + '/unknown')
    mkdir(input_directory + '/unknown')
    with open(input_directory + '/unknown/unknown01.txt', 'w+') as write_file:
        write_file.write(text)

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

    return predictions[0], decision_function[0][0]


@dataclass
class Node:
    def __init__(self, heuristic):
        self.parent = None
        self.children = []
        self.level = 0
        self.heuristic = heuristic
        self.id = 0
        self.text = ""
        self.predicted_author = ""
        self.score = 0.0

    def __eq__(self, other):
        if type(other) != Node:
            raise Exception("You cannot compare Node with type " + str(type(other)))
        if self.id == other.id:
            return True
        return False

    def __gt__(self, other):
        global author
        if type(other) != Node:
            raise Exception("You cannot compare Node with type " + str(type(other)))
        if self.predicted_author != author and other.predicted_author == author:
            return True
        if self.predicted_author == author and other.predicted_author != author:
            return False
        return self.score > other.score

    def __ge__(self, other):
        return self.__gt__(other) or self.score == other.score

    def __lt__(self, other):
        global author
        if type(other) != Node:
            raise Exception("You cannot compare Node with type " + str(type(other)))
        if self.predicted_author == author and other.predicted_author != author:
            return True
        if self.predicted_author != author and other.predicted_author == author:
            return False
        return self.score < other.score

    def __le__(self, other):
        return self.__lt__(other) or self.score == other.score


@dataclass
class StackInfo:
    def __init__(self, f_min, f_max):
        self.f_min = f_min
        self.f_max = f_max


class Phase:
    def __init__(self, original: Node):
        self.phase_number: int
        self.solutions: List[Node] = []
        self.original: Node = original
        self.heuristic_calls = {}
        self.heuristic_time = {}
        self.time_spent = 0
        self.phase_length = 0
        self.start_time = time()

    def best(self, heuristic):
        best_solution = None
        for solution in self.solutions:
            if best_solution is None:
                best_solution = solution
            elif solution > best_solution:
                best_solution = solution

        count = 0
        for solution in self.solutions:
            if solution.heuristic == heuristic and solution.predicted_author == best_solution.predicted_author and solution.score == best_solution.score:
                count += 1
        return count

    def improvements(self, heuristic):
        count = 0
        for solution in self.solutions:
            if solution.heuristic == heuristic and solution > self.original:
                count += 1
        return count

    def worse(self, heuristic):
        count = 0
        for solution in self.solutions:
            if solution.heuristic == heuristic and solution < self.original:
                count += 1
        return count



class Tree:
    def __init__(self, text, fitness_function):
        self.root = Node(None)
        self.size = 0
        self.width = 0
        self.depth = 0
        self.members: List[Node] = []
        self.fitness_function = fitness_function
        paraphraser = ParaphraseCorpus()
        symbols = SymbolReplacement()
        translator = Translation()
        british = BritishAmericanNormalization()
        fillers = FillerWords()
        self.heuristics = [change_contraction_distribution, synonym_substitution,
                           remove_parenthetical_phrases, remove_appositive_phrases, remove_discourse_markers,
                           apply, transform_equation, replace_numbers, replace_short_forms,
                           symbols.replace_symbols, obfuscate_uppercase, paraphraser.obfuscate,
                           obfuscate_stopwords, obfuscate_punctuation_count,
                           obfuscate_unique_words_count, obfuscate_sentence_length, obfuscate_spelling,
                           british.create_errors, fillers.insert_random]
        self.heuristics = [castro_masking, mihaylova_masking, rahgouy_masking]

        self.root.text = text
        predicted_author, score = self.fitness_function(self.root.text)
        self.root.score = round(1 - score, 3)
        self.root.predicted_author = predicted_author
        self.members.append(self.root)
        self.function_evaluations = 1
        self.solution_found = False

    def add_node(self, parent: Node, new_node: Node):
        level_count = {}
        node_parent = None

        max_id = 0
        for member in [self.root] + self.members:
            if node_parent is None:
                if member == parent:
                    node_parent = member
            max_id = max(max_id, member.id)
            if member.level in level_count:
                level_count[member.level] += 1
            else:
                level_count[member.level] = 1

        if node_parent is not None:
            new_node.parent = node_parent
            new_node.level = node_parent.level + 1
            new_node.id = max_id + 1
            node_parent.children.append(new_node)
            self.size += 1
            self.members.append(new_node)
            self.depth = max(new_node.level, self.depth)
            return new_node
        else:
            raise Exception("The parent was not found in the tree")

    def visualize(self, tree_path="", final_node: Node = None):
        global author
        e = Digraph(tree_path + "/" + author + "_search_tree")

        e.attr('node', shape='box')

        success_chain = []
        if final_node is not None:
            success_chain.append(self.root.id)
            while final_node is not None:
                success_chain.append(final_node.id)
                final_node = final_node.parent

        for member in self.members:
            if member.id == 0:
                label = "original"
            else:
                label = str(member.heuristic.__name__)

            if member.id in success_chain:
                e.node(str(member.id), label=label, color='lightblue2', style='filled')
            else:
                e.node(str(member.id), label=label)

            if member.parent is not None:
                e.edge(str(member.parent.id), str(member.id))
        e.render(format='svg')
        e.render(format='png')


class MCTS(Tree):
    def __init__(self, text, fitness_function):
        Tree.__init__(self, text, fitness_function)
        # self.max_tree_depth = len(self.heuristics) ** 2
        self.max_tree_depth = None
        self.max_tree_width = None
        self.memory_size = 12
        self.scaling_factor = 1.0
        self.average_reward = {}
        self.arm_pulls = {}
        self.memory: List[Node] = []
        self.average_reward[0] = self.root.score
        self.arm_pulls[0] = 0
        self.memory.append(self.root)

    def add_node(self, parent: Node, new_node: Node):
        new_node = Tree.add_node(self, parent, new_node)
        self.average_reward[new_node.id] = 0.0
        self.arm_pulls[new_node.id] = 0
        return new_node

    def visualize(self, tree_path=dirname(dirname(abspath(__file__))) + "/AMT_Results/MCTS/", final_node: Node = None,
                  visualize_memory=False):
        Tree.visualize(self, tree_path, final_node)
        if not visualize_memory:
            return
        global author
        e = Digraph(tree_path + "/" + author + "_memory")

        e.attr('node', shape='box')

        members = []
        for member in self.memory:
            if member.heuristic is None:
                label = str("original")
            else:
                label = str(member.heuristic.__name__)

            e.node(str(member.id), label=label)
            members.append(member.id)

        for member in self.memory:
            if member.parent is not None:
                if member.parent.id in members and member.id in members:
                    e.edge(str(member.parent.id), str(member.id))

        e.render(format='svg')
        e.render(format='png')

    def UCB(self, arm: Node):
        average_reward = self.average_reward[arm.id]
        arm_pulls = max(self.arm_pulls[arm.id], 1)
        return average_reward + self.scaling_factor * sqrt(2 * log(sum(self.arm_pulls.values())) / arm_pulls)

    def selection(self):
        selected = None

        max_ucb = -0.1
        level_count = {}

        if len(self.members) == 1:
            return self.root

        for member in self.members:
            if member.level in level_count:
                level_count[member.level] += 1
            else:
                level_count[member.level] = 1
            if level_count[member.level] == 1:
                level_count[member.level + 1] = 0

        for member in self.memory:
            ucb = self.UCB(member)

            replace = False
            if max_ucb == ucb:
                replace = randint(0, 1)

            if replace or max_ucb < ucb and len(member.children) < len(self.heuristics):
                if self.max_tree_width is None or level_count[member.level + 1] < self.max_tree_width:
                    if self.max_tree_depth is None or member.level < self.max_tree_depth:
                        max_ucb = ucb
                        selected = member
        return selected

    def expansion(self, selected_node: Node):
        set3 = list(set(self.heuristics) - set([child.heuristic for child in selected_node.children]))
        heuristic = set3[randint(0, len(set3) - 1) if len(set3) > 1 else 0]
        new_node = Node(heuristic)
        new_node.text = new_node.heuristic(selected_node.text)
        new_node = self.add_node(parent=selected_node, new_node=new_node)
        return new_node

    def simulation(self, new_node: Node):
        predicted_author, score = self.fitness_function(new_node.text)
        self.function_evaluations += 1
        reward = round(1 - score, 3)
        if len(self.memory) < self.memory_size:
            self.memory.append(new_node)
        else:
            member_to_replace = self.memory[randint(0, self.memory_size - 1) if self.memory_size > 1 else 0]
            if self.average_reward[member_to_replace.id] < reward:
                self.memory.remove(member_to_replace)
                self.memory.append(new_node)
            else:
                delta = self.average_reward[member_to_replace.id] - reward
                p = exp(-delta)
                r = random.random()
                if r < p:
                    self.memory.remove(member_to_replace)
                    self.memory.append(new_node)
        return predicted_author, reward

    def backpropagation(self, child_node: Node, reward: float):
        parent_node = child_node
        total_arm_pulls = 0
        while parent_node is not None:
            total_arm_pulls += self.arm_pulls[parent_node.id]
            self.average_reward[parent_node.id] = (self.average_reward[parent_node.id] *
                                                   total_arm_pulls + reward) / \
                                                  (total_arm_pulls + 1)
            if total_arm_pulls == 0:
                total_arm_pulls = 1
            parent_node = parent_node.parent
        self.arm_pulls[child_node.parent.id] += 1


    def search(self, iterations=10):
        new_node = None
        tree_path = dirname(dirname(abspath(__file__))) + "/AMT_Results/MCTS" + "_" + self.fitness_function.__name__ + "/"
        for iteration in range(iterations):
            if iteration == 0:
                node = self.root
            else:
                node = self.selection()

            if node is None:
                break

            new_node = self.expansion(node)
            predicted_author, reward = self.simulation(new_node)

            if not predicted_author.endswith(author):
                self.solution_found = True
                break
            self.backpropagation(new_node, reward)
        if self.solution_found:
            self.visualize(final_node=new_node)
        else:
            self.visualize()
        return new_node.text


class BeamStack(Tree):
    def __init__(self):
        Tree.__init__(self)
        self.beam_size = 3
        self.max_tree_depth = 3
        self.beam_stack = deque()
        self.open = {}
        self.closed = {}
        for index in range(self.max_tree_depth):
            for member in self.members:
                if member.level != index:
                    continue
                for heuristic in self.heuristics:
                    self.add_node(member, Node(heuristic))

    def generate_admitted_successors(self, node: Node, beam_stack_top: StackInfo):
        for child in node.children:
            child.text = child.heuristic(node.text)
            predicted_author, score = self.fitness_function(child.text)
            child.predicted_author = predicted_author
            child.score = score

        scores = self.f_scores(node.children)

        self.open[node.level + 1] = []
        for index, score in enumerate(scores):
            if beam_stack_top.f_min <= score < beam_stack_top.f_max:
                self.open[node.level + 1].append(node.children[index])

    def prune_layer(self, l: int):
        layer = self.open[l].copy()
        scores = self.f_scores(layer)
        for index in range(self.beam_size):
            min_node = layer[scores.index(min(scores))]
            scores.remove(min(scores))
            layer.remove(min_node)
        prune = layer
        self.beam_stack[0].f_max = min(scores)
        for node in prune:
            self.open[l].remove(node)
            self.members.remove(node)

    def f_scores(self, layer: list):
        scores = []
        for node in layer:
            scores.append(self.f(node))
        return scores

    def f(self, n: Node):
        return self.g(n) + self.h(n)

    def g(self, n: Node):
        return n.level * n.score

    def h(self, n: Node):
        return self.max_tree_depth - n.level

    def search(self, upper_bound: float, relay: int):
        self.open[0] = [self.root]
        self.open[1] = []
        self.closed[0] = []
        best_goal = None
        layer_index = 0
        while len(self.open[layer_index]) != 0 or len(self.open[layer_index + 1]) != 0:
            while len(self.open[layer_index]) != 0:
                layer = self.open[layer_index]
                scores = self.f_scores(layer)
                node = layer[scores.index(min(scores))]
                self.open[layer_index].remove(node)
                self.closed[layer_index].append(node)
                if node.predicted_author != author:
                    upper_bound = self.g(node)
                    best_goal = node

                self.generate_admitted_successors(node, self.beam_stack[0])

                if len(self.open[layer_index + 1]) > self.beam_size:
                    self.prune_layer(layer_index + 1)

            if 1 < layer_index <= relay or layer_index > relay + 1:
                for node in self.closed[layer_index - 1]:
                    self.closed[layer_index - 1].remove(node)

            layer_index += 1
            self.open[layer_index + 1] = []
            self.closed[layer_index] = []
            self.beam_stack.append(StackInfo(0, upper_bound))

        return best_goal

    def DCBSS(self, upper_bound: float, relay: int):

        self.beam_stack.append(StackInfo(0, upper_bound))
        optimal_solution = None
        while self.beam_stack[0] is not None:
            solution = self.search(upper_bound=upper_bound, relay=relay)
            if solution is not None:
                optimal_solution = solution
                upper_bound = self.g(solution)

            while self.beam_stack[0].f_max >= upper_bound:
                self.beam_stack.pop()
                if len(self.beam_stack) == 0:
                    return optimal_solution

            self.beam_stack[0].f_min = self.beam_stack[0].f_max
            self.beam_stack[0].f_max = upper_bound


class AdapHH(Tree):
    def __init__(self, text, fitness_function):
        Tree.__init__(self, text, fitness_function)
        self.phase_factor = 500
        self.phase_requested = 100
        self.phase_length = 0
        self.tabu_durations = {}
        self.heuristic_calls = {}
        self.heuristic_time = {}
        self.phases: List[Phase] = []
        for heuristic in self.heuristics:
            self.heuristic_time[heuristic] = 0
            self.heuristic_calls[heuristic] = 0
        self.tabu_upper_bound = 2 * sqrt(2 * len(self.heuristics))
        self.excluded_heuristics = []
        self.heuristic_probability = {}
        # weight for best result in current phase
        self.w1 = 1.0
        # weight for improvements in current phase
        self.w2 = 1.0
        # weight for worse solutions in current phase
        self.w3 = 1.0
        # weight for total improvements in all phases
        self.w4 = 1.0
        # weight for total worse solutions in all phases
        self.w5 = 1.0
        self.total_time = 10
        self.time_spent = 0

    def search(self):
        def p(phase: Phase, heuristic):
            return self.w1 * (((phase.best(heuristic) + 1) ** 2) * ((self.total_time - self.time_spent) / max(phase.heuristic_time[heuristic], 0.001)) * b(phase)) + \
                   self.w2 * (phase.improvements(heuristic) / max(phase.heuristic_time[heuristic], 0.001)) - self.w3 * (phase.worse(heuristic) / max(phase.heuristic_time[heuristic], 0.001)) + \
                   self.w4 * (sum([phase.improvements(heuristic) for phase in self.phases]) / max(self.heuristic_time[heuristic], 0.001)) - self.w5 * (sum([phase.worse(heuristic) for phase in self.phases]) / max(self.heuristic_time[heuristic], 0.001))

        def b(phase: Phase):
            return int(sum([phase.best(i) for i in self.heuristics]) > 0)

        def extreme_heuristic_exclusion(phase: Phase):
            nb = sum([phase.improvements(heuristic) for heuristic in self.heuristics])
            if not nb > 1:
                return

            def exc(heuristic):
                fastest_heuristic = list(self.heuristic_time.keys())[list(self.heuristic_time.values()).index(min(self.heuristic_time.values()))]
                return (self.heuristic_time[heuristic] / self.heuristic_calls[heuristic]) / \
                       (self.heuristic_time[fastest_heuristic] / self.heuristic_calls[fastest_heuristic])
            exc_values = {}
            for heuristic in self.heuristics:
                if heuristic not in self.excluded_heuristics:
                    exc_values[heuristic] = exc(heuristic)

            if not std(list(exc_values.values())) > 2.0:
                return

            average_exc = sum(exc_values.values()) / len(exc_values.values())
            for heuristic in exc_values:
                if exc_values[heuristic] > 2 * average_exc:
                    self.excluded_heuristics.append(heuristic)

        root_node = self.root
        new_node = None
        while self.time_spent < self.total_time:
            phase = Phase(root_node)
            if len(self.phases) > 0:
                quality_index = {}
                for heuristic in self.heuristics:
                    quality_index[heuristic] = p(phase, heuristic)
                average_quality = sum(quality_index.values()) / len(quality_index.values())
                for heuristic in self.heuristics:
                    if quality_index[heuristic] < average_quality:
                        self.excluded_heuristics.append(heuristic)
                        if heuristic not in self.tabu_durations:
                            self.tabu_durations[heuristic] = sqrt(2 * len(self.heuristics))
                        else:
                            self.tabu_durations[heuristic] += 1
                        if self.tabu_durations[heuristic] >= 2 * sqrt(2 * len(self.heuristics)):
                            self.heuristics.remove(heuristic)
                extreme_heuristic_exclusion(self.phases[-1])
            phase.phase_length = round(sqrt(2 * len(self.heuristics)))
            for heuristic in self.heuristics:
                phase.heuristic_time[heuristic] = 0
            self.phases.append(phase)

            while self.phases[-1].time_spent < self.phases[-1].phase_length:
                for heuristic in set(self.heuristics) - set(self.excluded_heuristics):
                    self.heuristic_probability[heuristic] = ((sum([phase.best(heuristic) for phase in self.phases]) + 1) /
                                                             self.heuristic_time[heuristic]) ** (1 + 3 * (self.total_time - self.time_spent)
                                                                                  / self.total_time ** 3)
                for heuristic in self.heuristic_probability:
                    self.heuristic_probability[heuristic] /= sum(self.heuristic_probability.values())

                random_probability = random()
                chosen_heuristic = None
                for heuristic in self.heuristic_probability:
                    if random_probability < self.heuristic_probability[heuristic]:
                        chosen_heuristic = heuristic
                        break
                    else:
                        random_probability -= self.heuristic_probability[heuristic]
                new_node = Node(chosen_heuristic)
                heuristic_start = time()
                new_node.text = new_node.heuristic(root_node.text)
                self.heuristic_time[chosen_heuristic] += time() - heuristic_start
                self.phases[-1].heuristic_time[chosen_heuristic] += time() - heuristic_start

                new_node.predicted_author, new_node.score = self.fitness_function(new_node.text)
                root_node = new_node
                self.heuristic_calls[chosen_heuristic] += 1
                self.phases[-1].time_spent += 1
                self.time_spent += 1

            self.excluded_heuristics.clear()
        return new_node.text


class PearlHunter(Tree):
    def __init__(self, text, fitness_function):
        Tree.__init__(self, text, fitness_function)
        self.num_of_snorkeling_times = len(self.heuristics)
        self.current_pool = self.root
        self.deep_dive_depth = 5
        self.move_history = []

    def random_initialization(self):
        self.current_pool = self.root

    def discover_environment_of_diving(self):
        return self.heuristics

    def select_moves(self):
        return self.heuristics

    def apply_move_to_pool(self, move):
        parent = self.current_pool
        pool = Node(move)
        pool.text = move(self.current_pool.text)
        pool.predicted_author, pool.score = self.fitness_function(pool.text)
        self.add_node(parent, pool)
        self.move_history.append(pool)
        self.current_pool = pool
        self.move_history.append(pool)
        return pool

    def trapped_around_buoy(self, pool):
        return self.move_history[-1].parent == pool.parent

    def snorkeling(self, environment, pool):
        best_dive = pool
        for heuristic in environment:
            shallow_dive = Node(heuristic)
            shallow_dive.text = heuristic(pool.text)
            shallow_dive.predicted_author, shallow_dive.score = self.fitness_function(pool.text)
            self.add_node(pool, shallow_dive)
            if shallow_dive > best_dive:
                best_dive = shallow_dive
        return best_dive

    def select_promising_positions(self, pools):
        promising_pool = None
        for pool in pools:
            if pool > promising_pool:
                promising_pool = pool
        return promising_pool

    def deep_dive(self, environment, pool):
        best_ever_found = None
        return best_ever_found

    def clear_pool(self):
        self.move_history.clear()
        pass

    def search(self):
        environment = self.discover_environment_of_diving()
        predicted_author = author
        best_ever_found = None
        while predicted_author == author:
            moves = self.select_moves()
            for move in moves:
                pools = set()
                for snorkeling_time in range(self.num_of_snorkeling_times):
                    pool = self.apply_move_to_pool(move)
                    while self.trapped_around_buoy(pool):
                        self.apply_move_to_pool(move)
                    pool = self.snorkeling(environment, pool)
                    pools.add(pool)
                optimal_pool = self.select_promising_positions(pools)
                best_ever_found = max(best_ever_found, self.deep_dive(environment, optimal_pool))
            if len(self.move_history) > 100:
                self.clear_pool()
                self.random_initialization()
            self.visualize()
            predicted_author = best_ever_found.predicted_author
        return best_ever_found


class ML:
    pass


class VNS(Tree):
    def __init__(self, text, fitness_function):
        Tree.__init__(self, text, fitness_function)
        self.local_search_iterations = 25
        self.heuristic_rank = {}
        self.population: List[Node] = []
        self.tabu = []
        while len(self.tabu) < len(self.heuristics) / 2:
            random_heuristic = self.heuristics[randint(0, len(self.heuristics) - 1)]
            if random_heuristic not in self.tabu:
                self.tabu.append(random_heuristic)
        self.population_size = 10
        for heuristic in self.heuristics:
            self.heuristic_rank[heuristic] = 0

    def shaking(self, x: Node):
        heuristic = self.tabu[randint(0, len(self.tabu) - 1)]
        new_node = Node(heuristic)
        new_node.text = new_node.heuristic(x.text)
        new_node.predicted_author, new_node.score = self.fitness_function(new_node.text)
        new_node.score = round(1 - new_node.score, 3)
        return new_node, heuristic

    def local_search(self, x: Node):
        non_improving_iteration = 0
        best_heuristics = []
        while max(self.heuristic_rank.values()) > -1:
            for heuristic in self.heuristic_rank:
                if self.heuristic_rank[heuristic] == max(self.heuristic_rank.values()):
                    best_heuristics.append(heuristic)

            max_heuristic = best_heuristics[randint(0, len(best_heuristics) - 1) if len(best_heuristics) > 1 else 0]
            new_node = Node(max_heuristic)
            new_node.text = new_node.heuristic(x.text)
            new_node.predicted_author, new_node.score = self.fitness_function(new_node.text)
            new_node.score = round(1 - new_node.score, 3)
            if new_node > x:
                non_improving_iteration = 0
                self.add_node(x, new_node)
                x = new_node
                for heuristic in self.heuristic_rank:
                    self.heuristic_rank[heuristic] = 1
            else:
                non_improving_iteration += 1
                if new_node < x:
                    self.heuristic_rank[max_heuristic] = -1
                else:
                    self.heuristic_rank[max_heuristic] = 0
            if non_improving_iteration == self.local_search_iterations:
                break
            best_heuristics.clear()

        return x

    def search(self, iterations=20):
        root = self.root
        for iteration in range(iterations):
            x, h_s = self.shaking(root)
            x = self.local_search(x)
            if x < root or random() < 0.2:
                self.tabu.insert(0, h_s)
                self.tabu.pop()
            if len(self.population) == self.population_size:
                if x >= root:
                    self.population.remove(root)
                else:
                    minimum_score = 1.00
                    worst_solution = None
                    for member in self.population:
                        if member.score < minimum_score:
                            minimum_score = member.score
                            worst_solution = member
                    self.population.remove(worst_solution)
            self.population.append(x)
            if len(self.population) > 1:
                individual_1 = self.population[randint(0, len(self.population) - 1)]
                individual_2 = self.population[randint(0, len(self.population) - 1)]
                root = individual_1 if individual_1 > individual_2 else individual_2
            else:
                root = self.population[0]
            if """ periodical adjustment """:
                pass

            global author
            for individual in self.population:
                if individual.predicted_author != author:
                    return individual.text

    def visualize(self, tree_path=dirname(dirname(abspath(__file__))) + "/AMT_Results/MCTS/", final_node: Node = None):
        Tree.visualize(self, tree_path, final_node)


class EPH(Tree):
    def __init__(self, text, fitness_function):
        Tree.__init__(self, text, fitness_function)
        self.solution_population_size = 10
        self.heuristic_population_size = len(self.heuristics)
        self.number_of_rounds = 3
        self.heuristic_population: List[Node] = []
        self.solution_population: List[Node] = []

    def search(self):
        pass

class DNN_Fooler():
    def __init__(self, text, attribution_system, population_size=10, target=0.01):
        Tree.__init__(self, text, attribution_system)
        self.original_text = text
        self.attribution_system = attribution_system
        self.population_size = population_size
        self.population = []
        self.fitnesses = []
        self.target = target
        self.words = len(text.split(' '))

    def language_model(self):
        from nltk.corpus import reuters
        from nltk import ngrams
        from collections import Counter, defaultdict

        # Create a placeholder for model
        model = defaultdict(lambda: defaultdict(lambda: 0))

        # Count frequency of co-occurance
        google = dirname(dirname(abspath(__file__))) + '/Google 1 Billion copy/'
        corpus_files = listdir(google)
        corpus_text = ""
        for file in corpus_files:
            print(file)
            with open(google + file, errors='ignore') as read_file:
                corpus_text += read_file.read() + ' '
        sentences = word_tokenize(corpus_text)
        for sentence in sentences:
            for w1, w2, w3 in ngrams(sentence, pad_right=True, pad_left=True, n=3):
                model[(w1, w2)][w3] += 1

        # Let's transform the counts to probabilities
        for w1_w2 in model:
            total_count = float(sum(model[w1_w2].values()))
            for w3 in model[w1_w2]:
                model[w1_w2][w3] /= total_count

        print(dict(model["today", "the"]))

    def perturb(self, text, n=10, k=5):
        return text

    def adversarial_attack(self, generations=10):
        # initial population generation
        for member in range(self.population_size):
            new_text = self.perturb(self.original_text)
            self.population.append(new_text)

        for generation in range(generations):
            predicted_author, score = self.attribution_system(new_text)
            self.fitnesses.append(score * (-1 if predicted_author.endswith(author) else 1))

            if max(self.fitnesses) >= self.target: # successful adversarial attack
                return self.population[self.fitnesses.index(max(self.fitnesses))]
            else:
                print(self.population[0])
                new_population = []
                normalized_fitnesses = []
                minimum_fitness = min(self.fitnesses)
                fitness_sum = sum(self.fitnesses)
                for fitness in self.fitnesses:
                    normalized_fitnesses.append((fitness - minimum_fitness) / fitness_sum)
                new_population.append(self.population[self.fitnesses.index(max(self.fitnesses))])
                for member in range(self.population_size):
                    random_number = random.random()
                    for index in range(self.population_size):
                        if random_number < normalized_fitnesses[index]:
                            random_number -= normalized_fitnesses[index]
                        else:
                            parent1 = self.population[index].split(' ')
                            break
                    random_number = random.random()
                    for index in range(self.population_size):
                        if random_number < normalized_fitnesses[index]:
                            random_number -= normalized_fitnesses[index]
                        else:
                            parent2 = self.population[index].split(' ')
                            break

                    child = ""
                    for word in range(self.words):
                        child += (parent1[word] if random.random() < 0.50 else parent2[word]) + ' '

                    new_population.append(self.perturb(child))
                self.population = new_population


class AIM_IT(Tree):
    def __init__(self, text, fitness_function):
        Tree.__init__(self, text, fitness_function)

    def search(self, generations=10):
        max_reward = -1.0
        for generation in range(generations):
            if generation == 0:
                parent_node = self.root

            castro_node = Node(castro_masking)
            mihaylova_node = Node(mihaylova_masking)
            rahgouy_node = Node(rahgouy_masking)

            castro_node.text = castro_node.heuristic(parent_node.text)
            mihaylova_node.text = mihaylova_node.heuristic(parent_node.text)
            rahgouy_node.text = rahgouy_node.heuristic(parent_node.text)

            self.add_node(parent_node, castro_node)
            self.add_node(parent_node, mihaylova_node)
            self.add_node(parent_node, rahgouy_node)

            predicted_author, castro_score = self.fitness_function(castro_node.text)
            castro_reward = castro_score * (-1 if predicted_author.endswith(author) else 1)

            predicted_author, mihaylova_score = self.fitness_function(mihaylova_node.text)
            mihaylova_reward = mihaylova_score * (-1 if predicted_author.endswith(author) else 1)

            predicted_author, rahgouy_score = self.fitness_function(rahgouy_node.text)
            rahgouy_reward = rahgouy_score * (-1 if predicted_author.endswith(author) else 1)

            if max_reward < max(castro_reward, mihaylova_reward, rahgouy_reward):
                max_reward = max(castro_reward, mihaylova_reward, rahgouy_reward)
            else:
                break

            if max_reward == castro_reward:
                parent_node = castro_node
            elif max_reward == mihaylova_reward:
                parent_node = mihaylova_node
            else:
                parent_node = rahgouy_node

            if max_reward > 0:
                self.solution_found = True
                self.visualize(dirname(dirname(abspath(__file__))) + "/AMT_Results/" + hyper_heuristic.__name__,
                               parent_node)
                return parent_node.text

        self.visualize(dirname(dirname(abspath(__file__))) + "/AMT_Results/" + hyper_heuristic.__name__)
        return self.root.text


if __name__ == '__main__':
    hyper_heuristic = MCTS
    fitness_function = LSVM
    max_iterations = [1, 2, 4, 8, 16, 32, 64, 128]
    max_iterations = [128, 256, 512, 1024]
    runs = 1
    num_authors = 25

    for run in range(runs):
        for max_it in max_iterations:
            function_evaluations = []
            print("Max Iterations: " + str(max_it))

            if not isdir(dirname(dirname(abspath(__file__))) + "/AMT_Results/" + hyper_heuristic.__name__ + "_" +
                         fitness_function.__name__):
                mkdir(dirname(dirname(abspath(__file__))) + "/AMT_Results/" + hyper_heuristic.__name__ + "_" +
                      fitness_function.__name__)
            elif len(listdir(dirname(dirname(abspath(__file__))) + "/AMT_Results/" + hyper_heuristic.__name__ + "_" +
                           fitness_function.__name__)) % num_authors == 0:
                rmtree(dirname(dirname(abspath(__file__))) + "/AMT_Results/" + hyper_heuristic.__name__ + "_" +
                       fitness_function.__name__)
                mkdir(dirname(dirname(abspath(__file__))) + "/AMT_Results/" + hyper_heuristic.__name__ + "_" +
                      fitness_function.__name__)
            for run_author in range(1000, 1000 + num_authors):
                author = str(run_author)
                if isfile(dirname(dirname(abspath(__file__))) + "/AMT_Results/" + hyper_heuristic.__name__ + "_" +
                          fitness_function.__name__ + "/" + author + "_4.txt"):
                    continue

                print(author)
                tree = hyper_heuristic(special_read(dirname(dirname(abspath(__file__))) +
                                                    "/CASIS Versions/CASIS-1000_Dataset - Unknown/" + author + "_4.txt"),
                                       fitness_function)
                solution = tree.search(max_it)
                with open(dirname(dirname(abspath(__file__))) + "/AMT_Results/" + hyper_heuristic.__name__ + "_" +
                          fitness_function.__name__ + "/" + author + "_4.txt", "w+") as masked:
                    masked.write(solution)

                if tree.solution_found:
                    function_evaluations.append(tree.function_evaluations)


            print("Function Evaluations: " + str(sum(function_evaluations) / len(function_evaluations) - 1 if len(function_evaluations) > 0 else 0))

            run_author_attribution(known_directory=dirname(dirname(abspath(__file__))) + '/CASIS Versions/CASIS-25_Dataset - Known/',
                                   unknown_directory=dirname(dirname(abspath(__file__))) + "/AMT_Results/" +
                                                     hyper_heuristic.__name__ + "_" + fitness_function.__name__ + "/",
                                   attribution_used=['lsvm', 'teahan', 'koppel'], verbose=True, runs=1)


            mkdir(dirname(dirname(abspath(__file__))) + '/AMT_Results/MCTS-1-' + str(max_it))
            move(dirname(dirname(abspath(__file__))) + '/AMT_Results/MCTS_LSVM/', dirname(dirname(abspath(__file__))) + '/AMT_Results/MCTS-1-' + str(max_it) + '/Samples')
            # move(dirname(dirname(abspath(__file__))) + '/AMT_Results/MCTS/', dirname(dirname(abspath(__file__))) + '/AMT_Results/MCTS-1-' + str(max_it) + '/Trees')
