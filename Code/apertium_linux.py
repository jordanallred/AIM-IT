from subprocess import Popen, PIPE
from shutil import copy
from os.path import isfile, isdir, join, abspath, dirname, relpath, realpath
from os import listdir, walk, getcwd


def run_spanish():
    ubuntu = None
    for user in listdir('C:/Users'):
        if not isdir('C:/Users/' + user):
            continue
        if isdir('C:/Users/' + user + '/AppData/Local/Microsoft/WindowsApps/'):
            for directory in listdir('C:/Users/' + user + '/AppData/Local/Microsoft/WindowsApps/'):
                if isdir(
                        'C:/Users/' + user + '/AppData/Local/Microsoft/WindowsApps/' + directory) and directory.lower().__contains__(
                        'ubuntu'):
                    ubuntu = 'C:/Users/' + user + '/AppData/Local/Microsoft/WindowsApps/' + directory + '/ubuntu.exe'
                if directory == 'ubuntu.exe':
                    ubuntu = 'C:/Users/' + user + '/AppData/Local/Microsoft/WindowsApps/ubuntu.exe'
                if ubuntu is not None:
                    break
        if ubuntu is not None:
            break

    if ubuntu is None:
        exe = "ubuntu.exe"
        for root, dirs, files in walk('C:/Users'):
            for name in files:
                if name == exe:
                    ubuntu = abspath(join(root, name))
                if ubuntu is not None:
                    break
            if ubuntu is not None:
                break

    process = Popen("\"" + ubuntu + "\"", stdin=PIPE, encoding='utf8')
    process.communicate('apertium ' + 'en' + '-' + 'es' +
                        ' "/mnt/c/' + dirname(dirname(abspath(__file__)))[3:].replace('\\', '/') + '/Code/Allred/input.txt"'
                        + ' "/mnt/c/' + dirname(dirname(abspath(__file__)))[3:].replace('\\', '/') + '/Code/Allred/spanish.txt"')

    process = Popen("\"" + ubuntu + "\"", stdin=PIPE, encoding='utf8')
    process.communicate('apertium ' + 'es' + '-' + 'en' +
                        ' "/mnt/c/' + dirname(dirname(abspath(__file__)))[3:].replace('\\', '/') + '/Code/Allred/spanish.txt"'
                        + ' "/mnt/c/' + dirname(dirname(abspath(__file__)))[3:].replace('\\', '/') + '/Code/Allred/output.txt"')
    copy(
        dirname(dirname(abspath(__file__))) + '/Code/Allred/output.txt',
        dirname(dirname(abspath(__file__))) + '/Code/Allred/spanish.txt')


def run_catalan():
    ubuntu = None
    for user in listdir('/Users'):
        if not isdir('/Users/' + user):
            continue
        if isdir('/Users/' + user + '/AppData/Local/Microsoft/WindowsApps/'):
            for directory in listdir('/Users/' + user + '/AppData/Local/Microsoft/WindowsApps/'):
                if isdir(
                        '/Users/' + user + '/AppData/Local/Microsoft/WindowsApps/' + directory) and directory.lower().__contains__(
                        'ubuntu'):
                    ubuntu = '/Users/' + user + '/AppData/Local/Microsoft/WindowsApps/' + directory + '/ubuntu.exe'
                if directory == 'ubuntu.exe':
                    ubuntu = '/Users/' + user + '/AppData/Local/Microsoft/WindowsApps/ubuntu.exe'
                if ubuntu is not None:
                    break
        if ubuntu is not None:
            break

    if ubuntu is None:
        exe = "ubuntu.exe"
        for root, dirs, files in walk('/Users'):
            for name in files:
                if name == exe:
                    ubuntu = abspath(join(root, name))
                if ubuntu is not None:
                    break
            if ubuntu is not None:
                break

    process = Popen("\"" + ubuntu + "\"", stdin=PIPE, encoding='utf8')
    process.communicate('apertium ' + 'en' + '-' + 'ca' +
                        ' "/mnt/c/' + dirname(dirname(abspath(__file__)))[3:].replace('\\', '/') + '/Code/Allred/input.txt"'
                        + ' "/mnt/c/' + dirname(dirname(abspath(__file__)))[3:].replace('\\', '/') + '/Code/Allred/catalan.txt"')

    process = Popen("\"" + ubuntu + "\"", stdin=PIPE, encoding='utf8')
    process.communicate('apertium ' + 'ca' + '-' + 'en' +
                        ' "/mnt/c/' + dirname(dirname(abspath(__file__)))[3:].replace('\\', '/') + '/Code/Allred/catalan.txt"'
                        + ' "/mnt/c/' + dirname(dirname(abspath(__file__)))[3:].replace('\\', '/') + '/Code/Allred/output.txt"')
    copy(
        dirname(dirname(abspath(__file__))) + '/Code/Allred/output.txt',
        dirname(dirname(abspath(__file__))) + '/Code/Allred/catalan.txt')


def run_gallegan():
    ubuntu = None
    exe = "ubuntu.exe"
    for root, dirs, files in walk('/Users/'):
        for name in files:
            if name == exe:
                ubuntu = abspath(join(root, name))
            if ubuntu is not None:
                break



    process = Popen("\"" + ubuntu + "\"", stdin=PIPE, encoding='utf8')
    process.communicate('apertium ' + 'en' + '-' + 'gl' +
                        ' "/mnt/c/' + dirname(dirname(abspath(__file__)))[3:].replace('\\', '/') + '/Code/Allred/input.txt"'
                        + ' "/mnt/c/' + dirname(dirname(abspath(__file__)))[3:].replace('\\', '/') + '/Code/Allred/gallegan.txt"')

    process = Popen("\"" + ubuntu + "\"", stdin=PIPE, encoding='utf8')
    process.communicate('apertium ' + 'gl' + '-' + 'en' +
                        ' "/mnt/c/' + dirname(dirname(abspath(__file__)))[3:].replace('\\', '/') + '/Code/Allred/gallegan.txt"'
                        + ' "/mnt/c/' + dirname(dirname(abspath(__file__)))[3:].replace('\\', '/') + '/Code/Allred/output.txt"')
    copy(
        dirname(dirname(abspath(__file__))) + '/Code/Allred/output.txt',
        dirname(dirname(abspath(__file__))) + '/Code/Allred/gallegan.txt')
