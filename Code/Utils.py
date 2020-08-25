from json import load, dump
from os import listdir, removedirs, remove, mkdir
from os.path import isdir, dirname, abspath, isfile
from shutil import copy, rmtree
from sys import executable, version
from random import randint
from math import ceil, floor
from Dictionary import effectiveness_coefficients


def find_python(python_version):
    if python_version != 3 and python_version != 2:
        raise Exception('Python ' + str(python_version) + ' does not exist.')
    if version[0] == str(python_version):
        return executable
    elif version[0] != str(python_version):
        folders = []
        path = 'C:/'
        if not isdir(path):
            path = '/'
        for file in listdir(path):
            if isdir(path + file) and not file.__contains__('$') and not file.__contains__('.'):
                folders.append(file.lower())
        result = search_for_python(python_version, path, folders)
        return result


def search_for_python(python_version, path, sub_folders):
    if len(sub_folders) == 0:
        return None
    for folder in sub_folders:
        if not isdir(path + folder):
            exit(-1)
        if str(folder).__contains__('python' + str(python_version)) or str(folder).__contains__(
                'anaconda' + str(python_version)):
            path += folder + '/python.exe'
            return path
    if 'users' in sub_folders:
        sub_folders.remove('users')
        sub_folders.insert(0, 'users')

    for folder in sub_folders:
        directories = []
        try:
            list_of_files = listdir(path + folder)
        except PermissionError:
            continue
        for option in list_of_files:
            if isdir(path + folder + '/' + option):
                directories.append(option.lower())
        result = search_for_python(python_version, path + folder + '/', directories)
        if result is not None:
            return result


def fix_encoding(file_directory, verbose=False):
    number = 1

    file_list = listdir(file_directory)
    for file in file_list:
        if verbose:
            print(file)
        if not file.__contains__('.txt'):
            continue
        path = file_directory + "/" + file
        with open(path, "r", encoding='utf8', errors='ignore') as open_file:
            data = open_file.read()
            new_data = data
            for letter in data:
                code = ord(letter)
                if code > 127 or code == 96:
                    if code == 195:
                        replace = 0
                    elif code == 175:
                        replace = 0
                    elif code == 194:
                        replace = 0
                    elif code == 162:
                        replace = 0
                    elif code == 163:
                        replace = 0
                    elif code == 176 or code == 186:
                        replace = 0
                    elif code == 187:
                        replace = 0
                    elif code == 191:
                        replace = 0
                    elif code == 239:
                        replace = 0
                    elif code == 8211 or code == 8212 or code == 8226:
                        replace = 45
                    elif code == 8216 or code == 8217 or code == 732:
                        replace = 39
                    elif code == 8220 or code == 8221:
                        replace = 34
                    elif code == 8230 or code == 166:
                        replace = 46
                    # apostrophe
                    elif code == 226:
                        replace = 0
                    elif code == 8364:
                        replace = 0
                    elif code == 8482:
                        replace = 39
                    elif code == 96:
                        replace = 39
                    else:
                        replace = 0
                    new_data = data
                    while new_data.__contains__(chr(code)):
                        new_data = new_data.replace(chr(code), chr(replace))
                    if code == replace:
                        print(letter)
                        print(code)
                        exit(-1)
                    else:
                        number += 1
        open_file.close()
        if not data == new_data:
            if verbose:
                print('changed')
            with open(path, "w", encoding='utf8') as open_file:
                open_file.write(new_data)
        else:
            if verbose:
                print('not changed')


def author_attribution_preprocessing(dataset_directory=None, validation_size=0.25, train_size=0.75, verbose=False,
                                     validation=False, train_directory=None, validation_directory=None, unknown_directory=None):

    input_directory = dirname(dirname(abspath(__file__))) + '/AAS_Input'
    if isdir(input_directory):
        rmtree(input_directory)

    mkdir(input_directory)
    mkdir(input_directory + '/unknown')

    if validation:
        mkdir(input_directory + '/validation')

    meta_data = {
        "folder": "unknown",
        "language": "EN",
        "encoding": "UTF8",
        "candidate-authors": [],
        "unknown-texts": []
    }

    unknown_mapping = {}

    if validation:
        train_size -= validation_size
        meta_data["validation-texts"] = []
        meta_data["validate"] = "validation"
    else:
        validation_size = 0.00

    ground_truth = {
        "ground-truth": []
    }

    validation_data = {
        "validation": []
    }

    texts_index = {}

    unknown_index = 1
    validation_index = 1

    if train_directory is not None:
        if not isdir(train_directory):
            train_directory = dirname(dirname(abspath(__file__))) + '/' + train_directory
        train_files = listdir(train_directory)
    else:
        train_files = []

    if validation_directory is not None:
        if not isdir(validation_directory):
            validation_directory = dirname(dirname(abspath(__file__))) + '/' + validation_directory
        validation_files = listdir(validation_directory)
    else:
        validation_files = []

    if unknown_directory is not None:
        if not isdir(unknown_directory):
            unknown_directory = dirname(dirname(abspath(__file__))) + '/' + unknown_directory
        unknown_files = listdir(unknown_directory)
    else:
        unknown_files = []

    if dataset_directory is None:
        search_list = train_files + validation_files + unknown_files
    else:
        search_list = listdir(dataset_directory)

    for directory in search_list:
        candidate = ""
        for letter in directory:
            if letter.isdigit():
                candidate += letter
            elif len(candidate) > 0:
                break
        if not directory.endswith('.txt'):
            continue
        candidate = 'candidate' + candidate

        if not isdir(input_directory + '/' + candidate):
            mkdir(input_directory + '/' + candidate)
            texts_index[candidate] = 1

        if {"author-name": candidate} not in meta_data["candidate-authors"]:
            meta_data["candidate-authors"].append({"author-name": candidate})

        if directory in train_files:
            copy(train_directory + '/' + directory,
                 input_directory + '/' + candidate + '/known' +
                 (str(texts_index[candidate]) if len(str(texts_index[candidate])) > 1 else '0' + str(texts_index[candidate])) + '.txt')
            texts_index[candidate] += 1
        elif directory in validation_files:
            validation_file = 'validation' + (str(validation_index) if len(str(validation_index)) > 1
                                              else '0' + str(validation_index)) + '.txt'
            copy(validation_directory + '/' + directory,
                 input_directory + "/validation/" + validation_file)
            meta_data["validation-texts"].append({"validation-text": validation_file})
            validation_data["validation"].append({"validation-text": validation_file,
                                                  "author": candidate})
            validation_index += 1
        elif directory in unknown_files:
            unknown_file = 'unknown' + (str(unknown_index) if len(str(unknown_index)) > 1
                                        else '0' + str(unknown_index)) + '.txt'
            unknown_mapping[unknown_file] = directory
            copy(unknown_directory + '/' + directory,
                 input_directory + '/unknown/' + unknown_file)
            meta_data["unknown-texts"].append({"unknown-text": unknown_file})
            ground_truth["ground-truth"].append({"unknown_text": unknown_file,
                                                 "author": candidate})
            unknown_index += 1
        else:
            raise IndexError

    with open(input_directory + "/" + 'meta-file.json', 'w+') as meta_json:
        dump(meta_data, meta_json)

    with open(input_directory + "/" + 'ground-truth.json', 'w+') as ground_truth_json:
        dump(ground_truth, ground_truth_json)

    if validation:
        with open(input_directory + "/" + 'validation.json', 'w+') as validation_json:
                dump(validation_data, validation_json)
    return train_files, validation_files, unknown_files, unknown_mapping


def cdaa_preprocessing(dataset_directory, verbose=False, train_size=0.75,
                       train_directory=None, unknown_directory=None):
    input_directory = dirname(dirname(abspath(__file__))) + '/AAS_Input'

    problem_directory = input_directory + '/problem00001'

    if isdir(input_directory):
        rmtree(input_directory)

    mkdir(input_directory)
    mkdir(problem_directory)
    mkdir(problem_directory + '/unknown')

    ground_truth = {
        "ground-truth": []
    }

    fandom_data = []

    problem_data = {
        "unknown-folder": "unknown",
        "candidate-authors": []
    }

    if train_directory is not None:
        if not isdir(train_directory):
            train_directory = dirname(dirname(abspath(__file__))) + '/' + train_directory
        train_files = listdir(train_directory)
    else:
        train_files = []

    if unknown_directory is not None:
        if not isdir(unknown_directory):
            unknown_directory = dirname(dirname(abspath(__file__))) + '/' + unknown_directory
        unknown_files = listdir(unknown_directory)
    else:
        unknown_files = []

    if dataset_directory is None:
        search_list = train_files  + unknown_files
    else:
        search_list = listdir(dataset_directory)

    texts_index = {}

    unknown_index = 1

    for directory in search_list:
        candidate = ""
        for letter in directory:
            if letter.isdigit():
                candidate += letter
            elif len(candidate) > 0:
                break
        if not directory.endswith('.txt'):
            continue
        candidate = 'candidate' + candidate

        if not isdir(problem_directory + '/' + candidate):
            mkdir(problem_directory + '/' + candidate)
            texts_index[candidate] = 1

        if {"author-name": candidate} not in problem_data["candidate-authors"]:
            problem_data["candidate-authors"].append({"author-name": candidate})

        if directory in train_files:
            copy(train_directory + '/' + directory,
                 problem_directory + '/' + candidate + '/known' +
                 (str(texts_index[candidate]) if len(str(texts_index[candidate])) > 1 else '0' + str(texts_index[candidate])) + '.txt')
            texts_index[candidate] += 1
        elif directory in unknown_files:
            unknown_file = 'unknown' + (str(unknown_index) if len(str(unknown_index)) > 1
                                        else '0' + str(unknown_index)) + '.txt'
            copy(unknown_directory + '/' + directory,
                 problem_directory + '/unknown/' + unknown_file)
            # problem_data["unknown_texts"].append({"unknown_text": unknown_file})
            ground_truth["ground-truth"].append({"unknown_text": unknown_file,
                                                 "author": candidate})
            unknown_index += 1
        else:
            raise IndexError

    if verbose:
        print('writing ground-truth')
    with open(problem_directory + "/" + 'ground-truth.json', 'w+') as ground_truth_json:
        dump(ground_truth, ground_truth_json)

    with open(problem_directory + "/fandom-info.json", 'w+') as meta_json:
        dump(fandom_data, meta_json)

    with open(problem_directory + "/problem-info.json", 'w+') as meta_json:
        dump(problem_data, meta_json)

    collection_data = [
        {
            "problem-name": "problem00001",
            "language": "en",
            "encoding": "UTF-8"
        }
    ]
    with open(input_directory + "/collection-info.json", 'w+') as meta_json:
        dump(collection_data, meta_json)
    return train_files, unknown_files


def author_verification_preprocessing(train_directory, output_directory=None):
    if output_directory is None:
        output_directory = dirname(dirname(abspath(__file__))) + '/AVS_Input'
    if isdir(output_directory):
        rmtree(output_directory)
    mkdir(output_directory)
    contents = {
        "language": "English",
        "problems": []
    }

    truth = ""

    if isdir(train_directory):
        print('reading known files')
        current_author = -1
        for file in listdir(train_directory):
            file_name = file[:-4]
            author = file_name.split('_')[0]
            candidate = "EN" + author

            if not isdir(output_directory + "/" + candidate):
                contents['problems'].append(candidate)
                mkdir(output_directory + "/" + candidate)
            copy(train_directory + "/" + file,
                 output_directory + "/" + candidate + "/known0" + str(len(listdir(output_directory + "/" + candidate))) + ".txt")

        for candidate in listdir(output_directory):
            for sample, file in enumerate(listdir(train_directory)):
                file_name = file[:-4]
                author = file_name.split('_')[0]

                copy(train_directory + "/" + file,
                     output_directory + "/" + candidate + "/unknown0" + str(sample) + ".txt")

                if candidate == "EN" + author:
                    truth += file + ' Y\n'
                else:
                    truth += file + ' N\n'


    print('writing contents')
    with open(output_directory + "/" + 'contents.json', 'w+') as contents_json:
        dump(contents, contents_json)

    print('writing truth')
    with open(output_directory + "/" + 'truth.txt', 'w+') as truth_file:
        truth_file.write(truth)


def get_answers(attribution_name, answers_file=None):
    if answers_file is None:
        answers_file = dirname(dirname(abspath(__file__))) + \
                     "/AAS_Results/" + attribution_name + "/answers.json"
    with open(answers_file) as json_file:
        data = load(json_file)

    answer_list = []
    try:
        for answer in data['answers']:
            answer_list.append(answer)
    except TypeError:
        for answer in data:
            answer_list.append(answer)
    return answer_list


def get_truth(truth_file=None):
    if truth_file is None:
        truth_file = dirname(dirname(abspath(__file__))) + '/AAS_Input/ground-truth.json'
    if not isfile(truth_file):
        truth_file = dirname(dirname(abspath(__file__))) + '/AAS_Input/problem00001/ground-truth.json'
    with open(truth_file, encoding='utf-8') as json_file:
        ground_truth_data = load(json_file)

    ground_truth = {}
    for file in ground_truth_data['ground-truth']:
        ground_truth.update({file['unknown_text']: file['author']})
    return ground_truth


def get_answers_accuracy(attribution_name, truth_file=None, answers_file=None):
    answers = get_answers(attribution_name, answers_file)
    ground_truth = get_truth(truth_file)

    correct = 0
    for answer in answers:
        if ground_truth[answer['unknown_text']] == answer['author']:
            correct += 1

    return 100 * correct / len(answers)


def fitness_function(answers, truth_file=None):
    if len(answers) == 0:
        raise Exception('Length of answers must be greater than zero')
    technique_list = list(answers.keys())
    ground_truth = get_truth(truth_file)
    author = 'author'
    unknown_text = 'unknown_text'

    fitness = {}
    for technique in technique_list:
        technique_answers = (answers[technique])
        for text in technique_answers:
            try:
                text[author]
            except KeyError:
                author = 'author'
                unknown_text = 'unknown_text'
            fitness[text[unknown_text]] = 0

    for technique in technique_list:
        technique_answers = (answers[technique])
        for text in technique_answers:
            if text[author] != ground_truth[text[unknown_text]]:
                if fitness[text[unknown_text]] > 0:
                    continue
                else:
                    fitness[text[unknown_text]] -= effectiveness_coefficients[technique] * text['score']
            else:
                if fitness[text[unknown_text]] < 0:
                    fitness[text[unknown_text]] = effectiveness_coefficients[technique] * text['score']
                else:
                    fitness[text[unknown_text]] += effectiveness_coefficients[technique] * text['score']
    return fitness


def get_obfuscation_mihaylova(results_directory):
    for file in listdir(results_directory):
        contents = ""
        if not isdir(results_directory + '/' + file):
            continue
        with open(results_directory + '/' + file + '/' + 'obfuscation.json',
                  'r') as obfuscation_json:
            obfuscation = load(obfuscation_json)
        for text in obfuscation:
            contents += text['obfuscation']
        with open(results_directory + '/' + file + '.txt',
                  'w+', encoding='utf8') as output_file:
            output_file.write(contents)
            rmtree(results_directory + '/' + file)
    # print(results_directory)
    # rmtree(results_directory)


def get_obfuscation_rahgouy(results_directory):
    for file in listdir(results_directory):
        if not isdir(results_directory + '/' + file):
            continue
        copy(results_directory + '/' + file + '/obfuscation.txt', results_directory + '/' + file + '.txt')
        if isdir(results_directory + '/' + file):
            rmtree(results_directory + '/' + file)
        elif isfile(results_directory + '/' + file):
            remove(results_directory + '/' + file)


def equalize_samples(input_directory):
    max_index = []
    for file in listdir(input_directory):
        index = -1
        for sample in listdir(input_directory + '/' + file):
            if sample.__contains__('('):
                if int(sample[sample.index('(') + 1]) > index:
                    index = int(sample[sample.index('(') + 1])
        max_index.append(index)
    max_index.sort()
    index = max_index[0]

    for file in listdir(input_directory):
        for sample in listdir(input_directory + '/' + file):
            if sample.__contains__('('):
                if int(sample[sample.index('(') + 1]) > index:
                    remove(input_directory + '/' + file + '/' + sample)

