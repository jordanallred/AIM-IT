from os import remove
from os.path import isfile, dirname, abspath
from random import randint
from shutil import copy
from subprocess import call
from time import time
from subprocess import Popen, PIPE


if __name__ == '__main__':
    author = 1
    text = 4
    verbose = 0
    epsilon = 0.00
    iterations = 5
    replacement_rate = 0.20
    replacement_loop_stop = 20

    error_file = dirname(dirname(abspath(__file__))) + "\\Code\\error.txt"
    directory = dirname(dirname(abspath(__file__))) + "\\Code\\"
    adversarial_directory = dirname(dirname(abspath(__file__))) + "\\Code\\Adversarial Text\\"
    input_file = dirname(dirname(abspath(__file__))) + "\\Code\\input.txt"
    best_file = dirname(dirname(abspath(__file__))) + "\\Code\\best_result.txt"
    language = dirname(dirname(abspath(__file__))) + "\\Code\\language.txt"
    original_file = dirname(dirname(abspath(__file__))) + "\\Code\\original.txt"
    pan_file = dirname(dirname(abspath(__file__))) + "\\Code\\PAN.txt"

    for author in range(25):
        try:
            test_file = "10" + (str(author) if author > 9 else "0" + str(author)) + "_" + str(text) + ".txt"

            if isfile(best_file):
                remove(best_file)
            if isfile(original_file):
                remove(original_file)
            if isfile(dirname(dirname(abspath(__file__))) + "\\Code\\gallegan.txt"):
                remove(dirname(dirname(abspath(__file__))) + "\\Code\\gallegan.txt")
            if isfile(dirname(dirname(abspath(__file__))) + "\\Code\\catalan.txt"):
                remove(dirname(dirname(abspath(__file__))) + "\\Code\\catalan.txt")
            if isfile(dirname(dirname(abspath(__file__))) + "\\Code\\spanish.txt"):
                remove(dirname(dirname(abspath(__file__))) + "\\Code\\spanish.txt")
            if isfile(dirname(dirname(abspath(__file__))) + "\\Code\\wordnet.txt"):
                remove(dirname(dirname(abspath(__file__))) + "\\Code\\wordnet.txt")

            if isfile(adversarial_directory + "adversarial_" + test_file):
                continue
            elif isfile(adversarial_directory + "best_" + test_file):
                # remove(adversarial_directory + "best_" + test_file)
                copy(adversarial_directory + "best_" + test_file, input_file)
                copy(adversarial_directory + "best_" + test_file, original_file)
                # continue

            for iteration in range(iterations):
                start = time()

                if isfile(error_file):
                    remove(error_file)
                if isfile(language):
                    remove(language)

                call(["C:\\Users\\Jordan Allred\\Anaconda3\\python.exe", dirname(dirname(abspath(__file__))) + "\\Code\\WordNet.py", str(replacement_rate), str(replacement_loop_stop)])
                call(["C:\\Users\\Jordan Allred\\Anaconda3\\python.exe", dirname(dirname(abspath(__file__))) + "\\Code\\Feature_Extractor.py", str(author), "wordnet", str(verbose)])

                # call(["C:\\Users\\Jordan Allred\\Anaconda3\\python.exe", dirname(dirname(abspath(__file__))) + "\\Code\\Apertium_Simpleton_Catalan.py"])

                process = Popen("\"C:\\Users\\Jordan Allred\\AppData\\Local\\Microsoft\\WindowsApps\\ubuntu.exe\"",
                                stdin=PIPE, encoding='utf8')
                process.communicate('apertium ' + 'en' + '-' + 'es' +
                                    ' "/mnt/c/Users/Jordan Allred/PycharmProjects/PAN/Code/AuthorCAATVI_src_07-18/scripts/apertium/input.txt'
                                    + '" "/mnt/c/Users/Jordan Allred/PycharmProjects/PAN/Code/AuthorCAATVI_src_07-18/scripts/apertium/spanish.txt')

                process = Popen("\"C:\\Users\\Jordan Allred\\AppData\\Local\\Microsoft\\WindowsApps\\ubuntu.exe\"",
                                stdin=PIPE, encoding='utf8')
                process.communicate('apertium ' + 'es' + '-' + 'en' +
                                    ' "/mnt/c/Users/Jordan Allred/PycharmProjects/PAN/Code/AuthorCAATVI_src_07-18/scripts/apertium/spanish.txt'
                                    + '" "/mnt/c/Users/Jordan Allred/PycharmProjects/PAN/Code/AuthorCAATVI_src_07-18/scripts/apertium/output.txt')
                copy(
                    'C:/Users/Jordan Allred/PycharmProjects/PAN/Code/AuthorCAATVI_src_07-18/scripts/apertium/output.txt',
                    'C:/Users/Jordan Allred/PycharmProjects/PAN/Code/AuthorCAATVI_src_07-18/scripts/apertium/spanish.txt')

                call(["C:\\Users\\Jordan Allred\\Anaconda3\\python.exe", dirname(dirname(abspath(__file__))) + "\\Code\\Feature_Extractor.py", str(author), "catalan", str(verbose)])

                # call(["C:\\Users\\Jordan Allred\\Anaconda3\\python.exe", dirname(dirname(abspath(__file__))) + "\\Code\\Apertium_Simpleton_Spanish.py"])

                process = Popen("\"C:\\Users\\Jordan Allred\\AppData\\Local\\Microsoft\\WindowsApps\\ubuntu.exe\"",
                                stdin=PIPE, encoding='utf8')
                process.communicate('apertium ' + 'en' + '-' + 'ca' +
                                    ' "/mnt/c/Users/Jordan Allred/PycharmProjects/PAN/Code/AuthorCAATVI_src_07-18/scripts/apertium/input.txt'
                                    + '" "/mnt/c/Users/Jordan Allred/PycharmProjects/PAN/Code/AuthorCAATVI_src_07-18/scripts/apertium/catalan.txt')

                process = Popen("\"C:\\Users\\Jordan Allred\\AppData\\Local\\Microsoft\\WindowsApps\\ubuntu.exe\"",
                                stdin=PIPE, encoding='utf8')
                process.communicate('apertium ' + 'ca' + '-' + 'en' +
                                    ' "/mnt/c/Users/Jordan Allred/PycharmProjects/PAN/Code/AuthorCAATVI_src_07-18/scripts/apertium/catalan.txt'
                                    + '" "/mnt/c/Users/Jordan Allred/PycharmProjects/PAN/Code/AuthorCAATVI_src_07-18/scripts/apertium/output.txt')
                copy(
                    'C:/Users/Jordan Allred/PycharmProjects/PAN/Code/AuthorCAATVI_src_07-18/scripts/apertium/output.txt',
                    'C:/Users/Jordan Allred/PycharmProjects/PAN/Code/AuthorCAATVI_src_07-18/scripts/apertium/catalan.txt')

                call(["C:\\Users\\Jordan Allred\\Anaconda3\\python.exe", dirname(dirname(abspath(__file__))) + "\\Code\\Feature_Extractor.py", str(author), "spanish", str(verbose)])

                # call(["C:\\Users\\Jordan Allred\\Anaconda3\\python.exe", dirname(dirname(abspath(__file__))) + "\\Code\\Apertium_Simpleton_Gallegan.py"])

                process = Popen("\"C:\\Users\\Jordan Allred\\AppData\\Local\\Microsoft\\WindowsApps\\ubuntu.exe\"",
                                stdin=PIPE, encoding='utf8')
                process.communicate('apertium ' + 'en' + '-' + 'gl' +
                                    ' "/mnt/c/Users/Jordan Allred/PycharmProjects/PAN/Code/AuthorCAATVI_src_07-18/scripts/apertium/input.txt'
                                    + '" "/mnt/c/Users/Jordan Allred/PycharmProjects/PAN/Code/AuthorCAATVI_src_07-18/scripts/apertium/gallegan.txt')

                process = Popen("\"C:\\Users\\Jordan Allred\\AppData\\Local\\Microsoft\\WindowsApps\\ubuntu.exe\"",
                                stdin=PIPE, encoding='utf8')
                process.communicate('apertium ' + 'gl' + '-' + 'en' +
                                    ' "/mnt/c/Users/Jordan Allred/PycharmProjects/PAN/Code/AuthorCAATVI_src_07-18/scripts/apertium/gallegan.txt'
                                    + '" "/mnt/c/Users/Jordan Allred/PycharmProjects/PAN/Code/AuthorCAATVI_src_07-18/scripts/apertium/output.txt')
                copy(
                    'C:/Users/Jordan Allred/PycharmProjects/PAN/Code/AuthorCAATVI_src_07-18/scripts/apertium/output.txt',
                    'C:/Users/Jordan Allred/PycharmProjects/PAN/Code/AuthorCAATVI_src_07-18/scripts/apertium/gallegan.txt')

                call(["C:\\Users\\Jordan Allred\\Anaconda3\\python.exe", dirname(dirname(abspath(__file__))) + "\\Code\\Feature_Extractor.py", str(author), "gallegan", str(verbose)])

                with open(error_file, 'r') as read_file:
                    content = read_file.read()
                score = content.split(',')

                method = best_method = score[0]
                error = best_error = score[1]

                if iteration == 0:
                    copy(error_file, best_error_file)
                    copy(directory + method + ".txt", input_file)
                else:
                    with open(best_error_file, 'r') as read_file:
                        content = read_file.read()
                    best_score = content.split(',')

                    best_error = best_score[1]
                    best_method = best_score[0]

                    if error > best_error and randint(0, 100) / 100 < epsilon or error < best_error:
                        best_method = method
                        best_error = error
                        if isfile(best_error_file):
                            remove(best_error_file)
                        copy(directory + method + ".txt", best_error_file)
                        copy(error_file, best_error_file)

                changed = False
                with open(input_file, 'r') as open_file:
                    content = open_file.read()
                    while content.__contains__("\n\n\n"):
                        changed = True
                        content = content.replace("\n\n\n", "\n\n")
                if isfile(input_file) and changed:
                    remove(input_file)
                    with open(input_file, 'w+') as open_file:
                        open_file.write(content)
                if isfile(best_file):
                    remove(best_file)
                copy(input_file, best_file)

                stop = time()

                print("author: " + str(author))
                print("iteration: " + str(iteration + 1))
                print("method: " + method)
                print("error: " + error)
                print("best method: " + best_method)
                print("best error: " + best_error)
                print("time: " + str(stop - start) + "s" + "\n")

                if float(best_error) < 0:
                    break
            if float(best_error) < 0:
                copy(best_file, adversarial_directory + "adversarial_" + test_file)
            else:
                copy(best_file, adversarial_directory + "best_" + test_file)
            # call(["C:\\Users\\Jordan Allred\\Anaconda3\\python.exe",
            # dirname(dirname(abspath(__file__))) + "\\Code\\Feature_Extractor.py", str(author), "original", str(1)])
            # call(["C:\\Users\\Jordan Allred\\Anaconda3\\python.exe", dirname(dirname(abspath(__file__))) + "\\Code\\Feature_Extractor.py", str(author), "best", str(1)])
            # call(["C:\\Users\\Jordan Allred\\Anaconda3\\python.exe", dirname(dirname(abspath(__file__))) + "\\Code\\Feature_Extractor.py", str(author), "pan", str(1)])
        except:
            continue
