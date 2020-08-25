from os import listdir, remove, mkdir
from os.path import isdir, dirname, abspath
from shutil import copy, rmtree
from multiprocessing import Pool
from multiprocessing import cpu_count
from multiprocessing import Manager
from Author_Masking import run_author_attribution, rahgouy_preprocessing, rahgouy_masking, mihaylova_masking, \
    castro_masking, allred_masking, get_obfuscation_mihaylova
from json import dump
from math import floor
from random import randint
from math import ceil
from time import time
from itertools import combinations
from sys import maxsize
import datetime


manager = Manager()
best_results, population, population_history, population_count, dataset = \
    manager.dict(), manager.dict(), manager.dict(), manager.dict(), manager.list()


def generate_intial_population(unknown_directory: str, output_directory: str):
    if isdir(output_directory):
        rmtree(output_directory)

    mkdir(output_directory)

    input_files = listdir(unknown_directory)
    for file in input_files:
        mkdir(output_directory + "/" + file.split('_')[0])
        copy(unknown_directory + "/" + file, output_directory + "/" + file.split('_')[0] + '/' +
             file.split('_')[0] + ".txt")
    return output_directory


def generation_driver(file):
    # print('\tWorking on Directory: ' + file)
    global search_directory

    search_directory = dirname(dirname(abspath(__file__))) + '/Adversarial Beam Search/'
    for author_masking_system in masks:
        if 'allred' == author_masking_system.lower():
            index = 0
            for search_file in listdir(search_directory):
                if search_file.__contains__('Allred'):
                    index += 1

            while True:
                try:
                    mkdir(search_directory + '/Allred(' + str(index) + ')')
                    break
                except FileExistsError:
                    index += 1
                    pass

            allred_masking(input_directory=search_directory + '/' + file, output_directory=search_directory + '/Allred(' + str(index) + ')',
                           remove_directory=False)

        if 'castro' == author_masking_system.lower():
            index = 0
            for search_file in listdir(search_directory):
                if search_file.__contains__('Castro'):
                    index += 1

            while True:
                try:
                    mkdir(search_directory + '/Castro(' + str(index) + ')')
                    break
                except FileExistsError:
                    index += 1
                    pass

            castro_masking(input_directory=search_directory + '/' + file, output_directory=search_directory + '/Castro(' + str(index) + ')', remove_directory=False)
        if 'mihaylova' == author_masking_system.lower():
            index = 0
            for search_file in listdir(search_directory):
                if search_file.__contains__('Mihaylova'):
                    index += 1

            while True:
                try:
                    mkdir(search_directory + '/Mihaylova(' + str(index) + ')')
                    break
                except FileExistsError:
                    index += 1
                    pass
            results_directory = mihaylova_masking(input_directory=search_directory + '/' + file, output_directory=search_directory + '/Mihaylova(' + str(index) + ')',
                                                  remove_directory=False)

            get_obfuscation_mihaylova(results_directory)

            for mihaylova_file in listdir(search_directory + '/' + file):
                copy(search_directory + '/' + file + '/' + mihaylova_file + '/original.txt', search_directory + '/' + file + '/' + mihaylova_file + '.txt')
                rmtree(search_directory + '/' + file + '/' + mihaylova_file)

        if 'rahgouy' == author_masking_system.lower():
            rahgouy_preprocessing(input_directory=search_directory + '/' + file, known_directory=dataset_directory + ' - Known')
            index = 0
            for search_file in listdir(search_directory):
                if search_file.__contains__('Rahgouy'):
                    index += 1

            while True:
                try:
                    mkdir(search_directory + '/Rahgouy(' + str(index) + ')')
                    break
                except FileExistsError:
                    index += 1
                    pass

            rahgouy_masking(input_directory=search_directory + '/' + file, known_directory=dataset_directory + ' - Known',
                            output_directory=search_directory + '/Rahgouy(' + str(index) + ')',
                            remove_directory=False)

            for rahgouy_directory in listdir(search_directory + '/' + file):
                copy(search_directory + '/' + file + '/' + rahgouy_directory + '/original.txt', search_directory + '/' + file + '/' + rahgouy_directory + '.txt')
                rmtree(search_directory + '/' + file + '/' + rahgouy_directory)


    for search_file in listdir(search_directory):
        if isdir(search_directory + '/' + search_file) and (search_file.__contains__('Rahgouy')
                    or search_file.__contains__('Allred') or search_file.__contains__('Castro')
                    or search_file.__contains__('Mihaylova')):
            my_file = False
            for working_file in listdir(search_directory + '/' + search_file):
                if working_file.__contains__(file):
                    my_file = True
                    break
            if my_file:
                for obfuscation_file in listdir(search_directory + '/' + search_file):
                    if isdir(search_directory + '/' + search_file + '/' + obfuscation_file):
                        for directory_file in listdir(search_directory + '/' + search_file + '/' + obfuscation_file):
                            if obfuscation_file.__contains__('('):
                                directory = obfuscation_file[:obfuscation_file.index('(')]
                            else:
                                directory = obfuscation_file
                            copy(search_directory + '/' + search_file + '/' + obfuscation_file + '/' + directory_file,
                                 search_directory + '/' + directory + '/' + directory + '(' +
                                 str(population_count[file]) + ').txt')
                            population_count[file] += 1
                        break
                    if obfuscation_file.__contains__('('):
                        directory = obfuscation_file[:obfuscation_file.index('(')]
                    else:
                        directory = obfuscation_file[:obfuscation_file.index('.')]
                    copy(search_directory + '/' + search_file + '/' + obfuscation_file, search_directory + '/' + directory + '/' + directory + '(' +
                         str(population_count[file]) + ').txt')
                    population_count[file] += 1
                rmtree(search_directory + '/' + search_file)
                break

    for index in range(len(attributions)):
        attributions[index] = attributions[index].lower()

    train_files, validation_files, unknown_files, input_directory = author_attribution_preprocessing(
        validation=False, train_directory=dataset_directory + ' - Known', unknown_directory=search_directory + '/' + file)

    results, fitness_score, unknown_mapping = run_author_attribution(attribution_used=attributions, preprocess=False,
                                                    train_directory=dataset_directory + ' - Train',
                                                    validation_directory=dataset_directory + ' - Validation',
                                                    known_directory=dataset_directory + ' - Known',
                                                    unknown_directory=search_directory + '/' + file,
                                                    input_directory=input_directory,
                                                    truth_file=input_directory + '/ground-truth.json',
                                                    output_directory=input_directory, runs=5)

    rmtree(input_directory)

    fitness = fitness_score
    population_list = listdir(search_directory + '/' + file)
    fitness_list = [fitness[key] for key in fitness.keys()]

    if len(population_history[file]) == 0:
        best_results[file] = min(fitness_list)
    elif min(fitness_list) < best_results[file]:
        best_results[file] = min(fitness_list)
    elif len(fitness_list) == beam_size:
        print("Stalled: " + file)
        dataset.remove(file)

    if best_results[file] < 0:
        print("Fooled: " + file)
        dataset.remove(file)

    if len(fitness_list) <= beam_size:
        population[file] = population_list
    else:
        for member in population_history[file]:
            if fitness_list.__contains__(member):
                index = fitness_list.index(member)
                del fitness_list[index]
                del population_list[index]

        for iteration in range(beam_size):
            index = fitness_list.index(min(fitness_list))
            population[file].append(population_list[index])
            del fitness_list[index]
            del population_list[index]
    population_history[file] += population[file]

    for search_file in listdir(search_directory + '/' + file):
        if search_file not in population[file]:
            remove(search_directory + '/' + file + '/' + search_file)


def initial_attribution(file):
    for index in range(len(attributions)):
        attributions[index] = attributions[index].lower()

    train_files, validation_files, unknown_files, input_directory = author_attribution_preprocessing(
        validation=False, train_directory=dataset_directory + ' - Known', unknown_directory=search_directory + '/' + file)

    results, fitness_score, unknown_mapping = run_author_attribution(attribution_used=attributions, preprocess=False,
                                                    input_directory=input_directory,
                                                    train_directory=dataset_directory + ' - Train',
                                                    validation_directory=dataset_directory + ' - Validation',
                                                    unknown_directory=search_directory + '/' + file,
                                                    known_directory=dataset_directory + ' - Known', runs=5,
                                                    truth_file=input_directory + '/ground-truth.json',
                                                    output_directory=input_directory)

    # (input_directory)

    fitness = fitness_score
    population_list = listdir(search_directory + '/' + file)
    fitness_list = [fitness[key] for key in fitness.keys()]

    if min(fitness_list) < 0:
        dataset.remove(file)

    best_results[file] = min(fitness_list)

    if len(fitness_list) <= beam_size:
        population[file] = population_list
    else:
        for member in population_history[file]:
            if fitness_list.__contains__(member):
                index = fitness_list.index(member)
                del fitness_list[index]
                del population_list[index]

        for iteration in range(beam_size):
            index = fitness_list.index(min(fitness_list))
            population[file].append(population_list[index])
            del fitness_list[index]
            del population_list[index]
    population_history[file] += population[file]

    for search_file in listdir(search_directory + '/' + file):
        if search_file not in population[file]:
            remove(search_directory + '/' + file + '/' + search_file)


def author_attribution_preprocessing(data_directory=None, validation_size=0.25, train_size=0.75, verbose=False,
                                     validation=False, train_directory=None, validation_directory=None,
                                     unknown_directory=None):
    if data_directory is None and train_directory is None and unknown_directory is None:
        raise Exception('Not enough directories given.')

    if data_directory is not None and \
            (train_directory is not None or unknown_directory is not None or validation_directory is not None):
        raise Exception('Too many directories given')

    search_files = listdir(search_directory)
    index = 0
    input_directory = search_directory + '/Author Attribution(' + str(index) + ')'
    for file in search_files:
        if file.__contains__('Attribution'):
            if isdir(input_directory):
                index += 1
                input_directory = search_directory + '/Author Attribution(' + str(index) + ')'
            else:
                break

    exists = False
    while not exists:
        try:
            mkdir(input_directory)
            exists = True

        except FileExistsError:
            exists = False
            index += 1
            input_directory = search_directory + '/Author Attribution(' + str(index) + ')'

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

    if len(train_files) == 0 and len(unknown_files) == 0:
        add_files = True
    else:
        add_files = False

    if data_directory is None:
        search_list = train_files + validation_files + unknown_files
    else:
        search_list = listdir(data_directory)
    for directory in search_list:
        candidate = ""
        for letter in directory:
            if letter.isdigit():
                candidate += letter
            elif len(candidate) > 0:
                break
        candidate = 'candidate' + candidate

        if not isdir(input_directory + '/' + candidate):
            mkdir(input_directory + '/' + candidate)
            texts_index[candidate] = 1

        meta_data["candidate-authors"].append({"author-name": candidate})

        if data_directory is not None:
            files_list = listdir(data_directory + '/' + directory)
            files_number = len(files_list)
            unknown_files += files_list.copy()
            train_number = max(floor(train_size * files_number), 1)
            if validation:
                validation_number = max(floor(validation_size * files_number) if validation else 0, 1)
            else:
                validation_number = 0

            if add_files:
                for number in range(train_number + validation_number):
                    random_number = randint(0, len(files_list) - 1)
                    if train_number > 0:
                        train_files.append(files_list[random_number])
                        train_number -= 1
                    else:
                        validation_files.append(files_list[random_number])
                    try:
                        unknown_files.remove(files_list[random_number])
                    except ValueError:
                        continue

            for file in files_list:
                if file in train_files:
                    copy(data_directory + '/' + directory + '/' + file,
                         input_directory + '/' + candidate + '/known' +
                         (str(texts_index[candidate])
                          if len(str(texts_index[candidate])) > 1 else '0' + str(texts_index[candidate])) + '.txt')
                    texts_index[candidate] += 1
                elif file in validation_files:
                    validation_file = 'validation' + (str(validation_index) if len(str(validation_index)) > 1
                                                      else '0' + str(validation_index)) + '.txt'
                    copy(data_directory + '/' + directory + '/' + file,
                         input_directory + "/validation/" + validation_file)
                    meta_data["validation-texts"].append({"validation-text": validation_file})
                    validation_data["validation"].append({"validation-text": validation_file,
                                                          "true-author": candidate})
                    validation_index += 1
                elif file in unknown_files:
                    unknown_file = 'unknown' + (str(unknown_index)
                                                if len(str(unknown_index)) > 1 else '0' + str(unknown_index)) + '.txt'
                    copy(data_directory + '/' + directory + '/' + file,
                         input_directory + '/unknown/' + unknown_file)
                    meta_data["unknown-texts"].append({"unknown-text": unknown_file})
                    ground_truth["ground-truth"].append({"unknown-text": unknown_file,
                                                         "true-author": candidate})
                    unknown_index += 1
                else:
                    raise IndexError
        else:
            if directory in train_files:
                copy(train_directory + '/' + directory,
                     input_directory + '/' + candidate + '/known' +
                     (str(texts_index[candidate]) if
                      len(str(texts_index[candidate])) > 1 else '0' + str(texts_index[candidate])) + '.txt')
                texts_index[candidate] += 1
            elif directory in validation_files:
                validation_file = 'validation' + (str(validation_index) if len(str(validation_index)) > 1
                                                  else '0' + str(validation_index)) + '.txt'
                copy(validation_directory + '/' + directory,
                     input_directory + "/validation/" + validation_file)
                meta_data["validation-texts"].append({"validation-text": validation_file})
                validation_data["validation"].append({"validation-text": validation_file,
                                                      "true-author": candidate})
                validation_index += 1
            elif directory in unknown_files:
                unknown_file = 'unknown' + (str(unknown_index) if len(str(unknown_index)) > 1
                                            else '0' + str(unknown_index)) + '.txt'
                if isdir(unknown_directory + '/' + directory):
                    for source_file in listdir(unknown_directory + '/' + directory):
                        copy(unknown_directory + '/' + directory + '/' + source_file,
                             input_directory + '/unknown/' + unknown_file)
                        meta_data["unknown-texts"].append({"unknown-text": unknown_file})
                        ground_truth["ground-truth"].append({"unknown-text": unknown_file,
                                                             "true-author": candidate})
                        unknown_index += 1

                else:
                    copy(unknown_directory + '/' + directory,
                         input_directory + '/unknown/' + unknown_file)
                    meta_data["unknown-texts"].append({"unknown-text": unknown_file})
                    ground_truth["ground-truth"].append({"unknown-text": unknown_file,
                                                         "true-author": candidate})
                    unknown_index += 1
            else:
                raise IndexError

    if verbose:
        ('writing meta-file')
    with open(input_directory + "/" + 'meta-file.json', 'w+') as meta_json:
        dump(meta_data, meta_json)

    if verbose:
        print('writing ground-truth')
    with open(input_directory + "/" + 'ground-truth.json', 'w+') as ground_truth_json:
        dump(ground_truth, ground_truth_json)

    if verbose:
        print('writing validation')
    if validation:
        with open(input_directory + "/" + 'validation.json', 'w+') as validation_json:
            dump(validation_data, validation_json)
    return train_files, validation_files, unknown_files, input_directory


def evolve_text(initialize=False):
    global dataset_directory
    global search_directory

    if not isdir(search_directory):
        mkdir(search_directory)

    from socket import gethostname
    if gethostname() == 'Jordan':
        pool = Pool(processes=2)
    else:
        pool = Pool(processes=ceil(cpu_count() * cpu_fraction))

    if dataset_directory is None:
        dataset_directory = 'CASIS Versions/CASIS-25_Dataset'
    if not isdir(dataset_directory):
        dataset_directory = dirname(dirname(abspath(__file__))) + '/' + dataset_directory

    print("Operating With " + str(ceil(cpu_count() * cpu_fraction)) + " CPUs")

    if initialize:
        generate_intial_population(unknown_directory=dataset_directory + ' - Unknown',
                                   output_directory=search_directory)

        global dataset
        for input_directory in dataset:
            population_history[input_directory] = []
            population_count[input_directory] = 0
            population[input_directory] = []

        print("Running Initial Attribution (" + str(len(dataset)) + " initial directories)")
        pool.map(initial_attribution, dataset)
        print()

    generation = 0
    while len(dataset) > 0:
        print("Running Generation No. " + str(generation) + ' (' + str(len(dataset)) + ' directories remaining)')
        start = time()
        pool.map(generation_driver, dataset)
        end = time()
        generation += 1
        if end - start < 60:
            print('Generation Execution Time = ' + str(round(end - start)) + ' seconds')
        elif end - start < 3600:
            print('Generation Execution Time = ' + str(round(end - start) // 60) + ' minutes and ' + str(round(end - start) % 60) + ' seconds')
        else:
            print('Generation Execution Time = ' + str(round(end - start) // 3600) + ' hours and ' + str((round(end - start) % 3600) // 60) + ' minutes and ' + str(round(end - start) % 60) + ' seconds')
        print(str(datetime.datetime.now()))
        print()


if __name__ == '__main__':
    skip_duplicates = True
    initialize = True
    mask_options = ['castro', 'mihaylova', 'rahgouy']
    attribution_options = ['LSVM']
    for index in range(1, len(attribution_options) + 1):
        for combination in combinations(attribution_options, index):
            message, name = "", ""
            name += '('
            message += "Running using "
            if len(mask_options) == 1:
                message += combination[0].capitalize()
            elif len(mask_options) == 2:
                message += combination[0].capitalize() + ' and ' + combination[1].capitalize()
            for mask in mask_options:
                if len(mask_options) > 2:
                    if mask_options.index(mask) < len(mask_options) - 2:
                        message += mask.capitalize() + ', '
                    elif mask_options.index(mask) < len(mask_options) - 1:
                        message += mask.capitalize() + ' and '
                    else:
                        message += mask.capitalize()
                name += mask.capitalize() + '_'
            name = name[:-1] + ') ('
            message += " masking with "
            if index == 1:
                message += combination[0].capitalize()
            elif index == 2:
                message += combination[0].capitalize() + ' and ' + combination[1].capitalize()
            for element in combination:
                if index > 2:
                    if combination.index(element) < index - 2:
                        message += element.capitalize() + ', '
                    elif combination.index(element) < index - 1:
                        message += element.capitalize() + ' and '
                    else:
                        message += element.capitalize()
                name += element.capitalize() + '_'
            name = name[:-1] + ')'
            message += ' attribution'

            cpu_fraction = 1
            search_directory = dirname(dirname(abspath(__file__))) + '/Adversarial Beam Search'

            masks = mask_options
            attributions = list(combination)
            dataset_directory = 'CASIS Versions/CASIS-25_Dataset'
            if not isdir(dataset_directory):
                dataset_directory = dirname(dirname(abspath(__file__))) + '/' + dataset_directory

            if not initialize:
                dataset = manager.list(listdir(search_directory))
            else:
                dataset = manager.list()
                for file in listdir(dataset_directory + ' - Unknown'):
                    dataset.append(file.split('_')[0])

            beam_size = 20

            if isdir(dirname(dirname(abspath(__file__))) + '/Adversarial Beam Search ' + name):
                if skip_duplicates:
                    continue
                rmtree(dirname(dirname(abspath(__file__))) + '/Adversarial Beam Search ' + name)

            print(message)
            print(name)
            evolve_text(initialize=initialize)

            from os import rename
            rename(dirname(dirname(abspath(__file__))) + '/Adversarial Beam Search',
                   dirname(dirname(abspath(__file__))) + '/Adversarial Beam Search ' + name)
    print('\nDONE')

    # Success!