import argparse
from json import load
from multiprocessing import Pool
from multiprocessing import cpu_count
from multiprocessing import Manager
from math import floor
from math import ceil
import datetime
from os import listdir, mkdir
from os import remove
from os.path import isdir, dirname, abspath, isfile
from shutil import copy, rmtree
from subprocess import call
from time import localtime, strftime
from time import time
from json import dump
from random import randint
from nltk import pos_tag
from nltk import word_tokenize
from nltk.corpus import wordnet
from nltk.corpus import wordnet as wn
from numpy.random import randint
from nltk.corpus import stopwords
from numpy import mean
from collections import Counter

manager = Manager()
best_results, population, population_history, population_count, dataset, mask_options, attribution_system = \
    manager.dict(), manager.dict(), manager.dict(), manager.dict(), manager.list(), manager.list(), manager.list()

effectiveness_coefficients = {
    'LSVM': 0.92,
    'Keselj': 0.56,
    'Teahan': 0.92,
    'Koppel': 0.80,
    'Benedetto': 0.60,
    'Stamatatos': 0.60,
    'Muttenthaler': 0.84,
    'Schaetti': 0.96,
    'Arun': 0.04,
    'Burrows': 0.44,
    'Gagala': 0.92,
}

coordinating_conjunctions = ['for', 'and', 'nor', 'but', 'or', 'yet', 'so']

subordinating_conjunctions = ['after', 'although', 'as', 'because', 'before', 'even if', 'even though', 'if',
                              'in order to', 'once', 'provided that', 'rather than', 'since', 'so that', 'than', 'that',
                              'though', 'unless', 'until', 'when', 'whenever', 'where', 'whereas', 'wherever',
                              'whether', 'which', 'while', 'who', 'whoever', 'whose', 'why']

conjunctive_adverbs = ['accordingly', 'also', 'besides', 'consequently', 'conversely', 'finally', 'furthermore',
                       'hence', 'however', 'indeed', 'instead', 'likewise', 'meanwhile', 'moreover', 'nevertheless',
                       'next', 'nonetheless', 'otherwise', 'similarly', 'still', 'subsequently', 'then', 'therefore',
                       'thus']

discourse_markers = ['additionally', 'at the same time',
                     'equally important', 'in addition', 'least of all',
                     'most of all', 'what\'s more', 'as a consequence of', 'as a result',
                     'as it is', 'as it was', 'due to', 'in response',
                     'as well as', 'equally', 'exactly', 'identically', 'in comparison', 'in much the same way',
                     'in relation to', 'like', 'matching', 'of little difference', 'parallel to', 'resembling',
                     'same as', 'similar to', 'a striking difference', 'accepting that', 'admittedly',
                     'after all', 'allowing that', 'and yet', 'another distinction',
                     'doubtless', 'for all that', 'fortunately', 'granted that',
                     'in another way',
                     'it is true that', 'it may well be', 'naturally', 'no doubt'
                     'of course', 'on one hand', 'on the other hand', 'on the contrary', 'opposite',
                     'to differ from', 'to oppose', 'unexpectedly', 'unfortunately',
                     'unlike', 'versus', 'without', 'while it is true', 'in conclusion', 'in short',
                     'last of all', 'to close', 'to conclude', 'to end', 'to recapitulate', 'to sum up', 'to summarize',
                     'against', 'be that as it may', 'by contrast', 'despite',
                     'despite this fact', 'even so', 'in contrast', 'in one way', 'in opposition to', 'in spite of',
                     'as a rule', 'broadly speaking', 'chiefly',
                     'essentially', 'generally', 'in essence', 'in general', 'in most cases', 'largely', 'mostly'
                     'on the whole', 'primarily', 'to some extent', 'usually', 'above all', 'absolutely', 'actually',
                     'as a matter of fact', 'clearly', 'certainly', 'each and every', 'especially',
                     'extremely', 'in truth', 'more and more', 'more importantly',
                     'most important of all', 'obviously', 'of major interest', 'significantly',
                     'surely', 'the chief characteristic', 'the climax of', 'the main issue', 'the major point',
                     'the most necessary', 'the most significant', 'to add to that', 'to be sure', 'to culminate',
                     'to emphasize', 'to highlight', 'to stress', 'undoubtedly', 'unquestionably', 'without a doubt',
                     'without question', 'as an example', 'as illustrated by', 'as revealed by', 'for example',
                     'for instance', 'in the case of', 'in this case',
                     'namely', 'such as', 'suppose that', 'to demonstrate', 'to explain', 'to illustrate', 'to show',
                     'specifically', 'to be exact', 'again', 'as stated', 'in fact', 'in other words', 'in particular',
                     'in simpler terms', 'that is to say', 'to clarify', 'to outline', 'to paraphrase',
                     'to put it in another way', 'to put it differently', 'to repeat', 'to rephrase', 'to restate',
                     'to review']

punctuations = [',', '.', '!', '?', ';', ':', '\"', '\'', ')']

stop_punctuations = ['.', '!', '?']

contractions = ["aren't", "can't", "could've", "couldn't", "couldn't've", "didn't", "doesn't", "don't", "e'er",
                "hadn't", "hasn't", "haven't", "he'd", "he'll", "he's", "how'd", "how'll", "how're", "how's", "I'd",
                "I'll", "I'm", "I've", "isn't", "it'd", "it'll", "it's", "let's", "might've", "must've", "ne'er",
                "o'clock", "o'er", "ol'", "oughtn't", "shan't", "she'd", "she'll", "she's", "should've", "shouldn't",
                "shouldn't've", "that'll", "that's", "that'd", "there'd", "there'll", "they'd", "they'll", "they're",
                "they've", "'tis", "'twas", "wasn't", "we'd", "we'd've", "we'll", "we're", "we've", "weren't", "what'd",
                "what'll", "what're", "what's", "what've", "when's", "where'd", "where's", "where've", "who'd",
                "who'd've", "who'll", "who's", "why'd", "why're", "why's", "won't", "would've", "wouldn't", "y'all",
                "you'd", "you'll", "you're", "you've", ]

expansions = ["are not", "can not", "could have", "could not", "could not have", "did not", "does not", "do not",
              "ever", "had not", "has not", "have not", "he would", "he will", "he is", "how would", "how will",
              "how are", "how is", "I would", "I will", "I am", "I have", "is not", "it would", "it will", "it is",
              "let us", "might have", "must have", "never", "of the clock", "over", "old", "ought not", "shall not",
              "she would", "she will", "she is", "should have", "should not", "should not have", "that will", "that is",
              "that would", "there would", "there will", "they would", "they will", "they are", "they have", "it is",
              "it was", "was not", "we would", "we would have", "we will", "we are", "we have", "were not",
              "what would", "what will", "what are", "what is", "what have", "when is", "where would", "where is",
              "where have", "who would", "who would have", "who will", "who is", "why would", "why are", "why is",
              "would not", "would have", "would not", "you all", "you would", "you will", "you are", "you have", ]

stop_words = list(stopwords.words('english'))

pos_dictionary = {
    'CC': 'coordinating conjunction',
    'CD': 'cardinal digit',
    'DT': 'determiner',
    'EX': 'existential',
    'FW': 'foreign word',
    'IN': 'preposition/subordinating conjunction',
    'JJ': 'adjective',
    'JJR': 'comparative adjective',
    'JJS': 'superlative adjective',
    'LS': 'list marker',
    'MD': 'modal',
    'NN': 'singular noun',
    'NNS': 'plural noun',
    'NNP': 'singular proper noun',
    'NNPS': 'plural proper noun',
    'PDT': 'predeterminer',
    'POS': 'possessive ending',
    'PRP': 'personal pronoun',
    'PRP$': 'possessive pronoun',
    'RB': 'adverb',
    'RBR': 'comparative adverb',
    'RBS': 'superlative adverb',
    'RP': 'particle',
    'TO': 'directional "to"',
    'UH': 'interjection',
    'VB': 'verb',
    'VBD': 'past tense verb',
    'VBG': 'present participle verb',
    'VBN': 'past participle verb',
    'VBP': 'present tense first person verb',
    'VBZ': 'present tense third person verb',
    'WDT': '"wh" determiner',
    'WP': '"wh" pronoun',
    'WP$': 'possessive "wh" pronoun',
    'WRB': '"wh" adverb'
}

british_words = ['flat', 'appetizer', 'fringe', 'hairslide', 'grill', 'grill', 'sweet', 'mobile phone', 'crisps',
                 'snakes and ladders', 'wardrobe', 'biscuit', 'candyfloss', 'anticlockwise', 'cot', 'nappy', 'chemist',
                 'aubergine', 'junior school, primary school', 'lift', 'motorway', 'chips', 'dustbin', 'petrol',
                 'bonnet', 'skipping rope', 'number plate', 'off-licence', 'postbox', 'oven glove', 'dummy', 'trousers',
                 'tights', 'car park', 'parting', 'full stop', 'public school', 'state school', 'dressing gown',
                 'shopping trolley', 'pavement', 'sledge', 'trainers', 'football', 'hundreds and thousands',
                 'pushchair', 'underground', 'braces', 'jumper', 'takeaway', 'drawing pin', 'noughts and crosses',
                 'boot', 'indicator', 'vest', 'holiday', 'waistcoat', 'flannel', 'postcode', 'courgette', 'centre',
                 'fibre', 'litre', 'theatre', 'colour', 'flavour', 'humour', 'labour', 'neighbour', 'apologise',
                 'organise', 'recognise', 'analyse', 'breathalyse', 'paralyse', 'leukaemia', 'manoeuvre', 'oestrogen',
                 'paediatric', 'defence', 'licence', 'offence', 'pretence', 'analogue', 'catalogue', 'dialogue',
                 'accommodation', 'action replay', 'aerofoil', 'aeroplane', 'agony aunt', 'Allen key', 'aluminium',
                 'aniseed', 'articulated lorry', 'asymmetric bars', 'baking tray', 'bank holiday', 'beetroot',
                 'black economy', 'blanket bath', 'block of flats', 'boiler suit', 'boob tube', 'bottom drawer',
                 'bowls', 'brawn ', 'breakdown van', 'breeze block', 'bridging loan', 'bumbag', 'casualty', 'catapult',
                 'central reservation', 'cinema', 'the movies', 'movie', 'cling film', 'common seal',
                 'consumer durables', 'cornflour', 'cos ', 'cot death', 'cotton bud', 'cotton wool', 'council estate',
                 'court card', 'crash barrier', 'crocodile clip', 'cross-ply', 'crotchet ', 'current account',
                 'danger money', 'demister ', 'dialling tone', 'diamante', 'double cream', 'draughts ', 'drink-driving',
                 'drinks cupboard', 'drinks party', 'dual carriageway', 'dust sheet', 'earth ', 'engaged',
                 'estate agent', 'estate agent', 'estate car', 'ex-directory', 'faith school', 'financial year',
                 'fire brigade', 'fire brigade', 'fire service', 'fire service', 'first floor', 'fish finger',
                 'fitted carpet', 'flexitime', 'flick knife', 'flyover', 'footway', 'full stop ', 'yard', 'garden',
                 'gearing ', 'gear lever', 'goods train', 'greaseproof paper', 'greaseproof paper', 'green fingers',
                 'ground floor', 'groundsman', 'hatstand', 'hen night', 'hire purchase', 'hoarding', 'hob', 'holdall',
                 'holidaymaker', 'homely', 'hosepipe', 'in hospital', 'hot flush', 'housing estate', 'ice lolly',
                 'icing sugar', 'indicator ', 'inside leg', 'jelly babies', 'Joe Bloggs', 'Joe Public', 'jumble sale',
                 'jump lead', 'kennel', 'ladybird', 'a lettuce', 'level crossing', 'lolly', 'lollipop lady',
                 'loose cover', 'lorry', 'loudhailer', 'low loader', 'lucky dip', 'luggage van', 'maize', 'mangetout',
                 'market garden', 'marshalling yard', 'maths', 'metalled road', 'milometer', 'minim  ', 'mobile phone',
                 'monkey tricks', 'motorway', 'motorway', 'mum', 'mum', 'mummy', 'mummy', 'nappy', 'needlecord',
                 'newsreader', 'noughts and crosses', 'number plate', 'off-licence', 'off-licence', 'opencast mining',
                 'ordinary share', 'oven glove', 'paddling pool', 'paracetamol', 'parting ', 'patience', 'pavement',
                 'pay packet', 'pedestrian crossing', 'peg', 'pelmet', 'petrol', 'petrol', 'physiotherapy',
                 'pinafore dress', 'plain chocolate', 'plain flour', 'polo neck', 'positive discrimination',
                 'postal vote', 'postbox', 'postcode', 'potato crisp', 'power point', 'pram', 'pram', 'press stud',
                 'press-up', 'private soldier', 'public school', 'public transport', 'punchbag', 'pushchair', 'pylon',
                 'quantity surveyor', 'quaver ', 'queue', 'racing car', 'railway', 'real tennis', 'recorded delivery',
                 'registration plate', 'remould ', 'reverse the charges', 'reversing lights', 'right-angled triangle',
                 'ring road', 'roundabout ', 'roundabout ', 'rowing boat', 'sailing boat', 'saloon ', 'sandpit',
                 'sandwich cake', 'sanitary towel', 'self-raising flour', 'semibreve ', 'semitone ', 'share option',
                 'shopping trolley', 'show home', 'show house', 'silencer ', 'silverside', 'skeleton in the cupboard',
                 'skimmed milk', 'skipping rope', 'skirting board', 'sledge', 'sleeper', 'sleeping partner',
                 'slowcoach', 'snakes and ladders', 'solicitor', 'soya', 'soya bean', 'splashback', 'spring onion',
                 'stag night', 'Stanley knife', 'starter', 'state school', 'storm in a teacup', 'surtitle', 'swede',
                 'sweets', 'takeaway ', 'takeaway ', 'taxi rank', 'tea towel', 'terrace house', 'tick', 'ticket tout',
                 'timber', 'titbit', 'toffee apple', 'touch wood', 'trade union', 'trading estate', 'trainers',
                 'transport cafe', 'trolley', 'twelve-bore', 'underground', 'vacuum flask', 'verge ', 'vest',
                 'veterinary surgeon', 'wagon ', 'waistcoat', 'walking frame', 'wardrobe', 'water ice', 'weatherboard',
                 'white coffee', 'white spirit', 'wholemeal bread', 'windcheater', 'windscreen', 'wing ', 'worktop',
                 'zebra crossing', 'zed ', 'zip']

american_words = ['apartment', 'starter', 'bangs', 'barrette', 'broil', 'broiler', 'candy', 'cell phone', 'chips',
                  'chutes and ladders', 'closet', 'cookie, cracker', 'cotton candy', 'counter clockwise', 'crib',
                  'diaper', 'drugstore', 'eggplant', 'elementary school', 'elevator', 'expressway, highway',
                  'French fries', 'garbage can', 'gas, gasoline', 'hood ', 'jump rope', 'license plate', 'liquor store',
                  'mailbox', 'oven mitt', 'pacifier', 'pants', 'pantyhose', 'parking lot', 'part ', 'period ',
                  'private school', 'public school', 'robe, bathrobe', 'shopping cart', 'sidewalk', 'sled', 'sneakers',
                  'soccer', 'sprinkles ', 'stroller', 'subway', 'suspenders', 'sweater', 'takeout ', 'thumbtack',
                  'tic-tac-toe', 'trunk ', 'turn signal ', 'undershirt', 'vacation', 'vest', 'washcloth', 'zip code',
                  'zucchini', 'center', 'fiber', 'liter', 'theater', 'color', 'flavor', 'humor', 'labor', 'neighbor',
                  'apologize', 'organize', 'recognize', 'analyze', 'breathalyze', 'paralyze', 'leukemia', 'maneuver',
                  'estrogen', 'pediatric', 'defense', 'license', 'offense', 'pretense', 'analog', 'catalog', 'dialog',
                  'accommodations', 'instant replay', 'airfoil', 'airplane', 'advice columnist', 'Allen wrench',
                  'aluminum', 'anise', 'tractor-trailer', 'uneven bars', 'cookie sheet', 'legal holiday', 'beet',
                  'underground economy', 'sponge bath', 'apartment building', 'coveralls', 'tube top', 'hope chest',
                  'lawn bowling', 'headcheese', 'tow truck', 'cinder block', 'bridge loan', 'fanny pack',
                  'emergency room', 'slingshot', 'median strip', 'movie', 'cinema', ' theater', 'plastic wrap',
                  'harbor seal', 'durable goods', 'cornstarch', 'Romaine', 'crib death', 'cotton swab',
                  'absorbent cotton', ' project', 'face card', 'guardrail', 'alligator clip', 'bias-ply',
                  'quarter note', 'checking account', 'hazard pay', 'defroster', 'dial tone', 'rhinestone',
                  'heavy cream', 'checkers', 'drunk driving', 'liquor cabinet', 'cocktail party', 'divided highway',
                  'drop cloth', 'ground', 'busy', 'real estate agent', 'realtor ', 'station wagon', 'unlisted',
                  'parochial school', 'fiscal year', 'fire department', 'fire company', 'fire department',
                  'fire company', 'second floor', 'fish stick', 'wall-to-wall carpeting', 'flextime', 'switchblade',
                  'overpass', 'sidewalk', 'period', 'lawn', 'lawn', 'leverage', 'gearshift', 'freight train',
                  'wax paper', 'waxed paper', 'green thumb', 'first floor', 'groundskeeper', 'hatrack',
                  'bachelorette party', 'installment plan', 'billboard', 'stovetop', 'carryall', 'vacationer', 'homey',
                  'hose', 'in the hospital', 'hot flash', 'housing development', 'Popsicle ', 'confectioners’ sugar',
                  'turn signal', 'inseam', 'jelly beans', 'Joe Blow', 'John Q. Public', 'rummage sale', 'jumper cable',
                  'doghouse', 'ladybug', 'a head of lettuce', 'grade crossing', 'lollipop', 'crossing guard',
                  'slipcover', 'truck', 'bullhorn', 'flatbed truck', 'grab bag', 'baggage car', 'corn', 'snow pea',
                  'truck farm', 'railroad yard', 'math', 'paved road', 'odometer', 'half note', 'cell phone',
                  'monkeyshines', 'highway', 'expressway', 'mommy', 'mom', 'mommy', 'mom', 'diaper', 'pinwale',
                  'newscaster', 'tic-tac-toe', 'license plate', 'package store', 'liquor store', 'open-pit mining',
                  'common stock', 'oven mitt', 'wading pool', 'acetaminophen', 'part', 'solitaire', 'sidewalk',
                  'pay envelope', 'crosswalk', 'clothespin', 'valance', 'gas', 'gasoline', 'physical therapy', 'jumper',
                  'dark chocolate', 'all-purpose flour', 'turtleneck', 'reverse discrimination', 'absentee ballot',
                  'mailbox', 'zip code', 'potato chip', 'electrical outlet', 'baby carriage', 'stroller', 'snap',
                  'pushup', 'GI', 'private school', 'public transportation', 'punching bag', 'stroller', 'utility pole',
                  'estimator', 'eighth note', 'line', 'race car', 'railroad', 'court tennis', 'certified mail',
                  'license plate', 'retread', 'call collect', 'back-up lights', 'right triangle', 'beltway', 'carousel',
                  'traffic circle', 'rowboat', 'sailboat', 'sedan', 'sandbox', 'layer cake', 'sanitary napkin',
                  'self-rising flour', 'whole note', 'half step', 'stock option', 'shopping cart', 'model home',
                  'model home', 'muffler', 'rump roast', 'skeleton in the closet', 'skim milk', 'jump rope',
                  'baseboard', 'sled', 'railroad tie', 'silent partner', 'slowpoke', 'chutes and ladders', 'lawyer',
                  'soy', 'soybean', 'backsplash', 'scallion', 'bachelor party', 'utility knife', 'appetizer',
                  'public school', 'tempest in a teapot', 'supertitle', 'rutabaga', 'candy', 'to go', 'takeout',
                  'taxi stand', 'dish towel', 'row house', 'check mark', 'scalper', 'lumber', 'tidbit',
                  'candy apple or caramel apple', 'knock on wood', 'labor union', 'industrial park', 'sneakers',
                  'truck stop', 'shopping cart', 'twelve-gauge', 'subway', 'thermos bottle', 'shoulder', 'undershirt',
                  'veterinarian', 'car', 'vest', 'walker', 'closet', 'Italian ice', 'clapboard', 'coffee with cream',
                  'mineral spirits', 'wholewheat bread', 'windbreaker', 'windshield', 'fender', 'countertop',
                  'crosswalk', 'zee', 'zipper']

minimal_set = {
    'keselj': {
        '1000_4': None,
        '1001_4': None,
        '1002_4': None,
        '1003_4': None,
        '1004_4': None,
        '1005_4': None,
        '1006_4': None,
        '1007_4': None,
        '1008_4': None,
        '1009_4': None,
        '1010_4': None,
        '1011_4': None,
        '1012_4': None,
        '1013_4': None,
        '1014_4': None,
        '1015_4': None,
        '1016_4': None,
        '1017_4': None,
        '1018_4': None,
        '1019_4': None,
        '1020_4': 'castro',
        '1021_4': None,
        '1022_4': None,
        '1023_4': None,
        '1024_4': None
    },
    'teahan': {
        '1000_4': 'rahgouy',
        '1001_4': None,
        '1002_4': None,
        '1003_4': None,
        '1004_4': 'rahgouy',
        '1005_4': None,
        '1006_4': None,
        '1007_4': None,
        '1008_4': ['allred', 'rahgouy'],
        '1009_4': None,
        '1010_4': ['allred', 'rahgouy'],
        '1011_4': None,
        '1012_4': None,
        '1013_4': None,
        '1014_4': ['allred', 'rahgouy'],
        '1015_4': None,
        '1016_4': None,
        '1017_4': 'allred',
        '1018_4': 'rahgouy',
        '1019_4': None,
        '1020_4': None,
        '1021_4': ['allred', 'rahgouy'],
        '1022_4': 'rahgouy',
        '1023_4': None,
        '1024_4': None
    },
    'koppel': {
        '1000_4': 'castro',
        '1001_4': ['allred', 'castro', 'rahgouy'],
        '1002_4': None,
        '1003_4': None,
        '1004_4': None,
        '1005_4': 'allred',
        '1006_4': None,
        '1007_4': None,
        '1008_4': None,
        '1009_4': None,
        '1010_4': 'rahgouy',
        '1011_4': None,
        '1012_4': ['allred', 'castro', 'rahgouy'],
        '1013_4': None,
        '1014_4': ['allred', 'castro', 'rahgouy'],
        '1015_4': None,
        '1016_4': None,
        '1017_4': None,
        '1018_4': None,
        '1019_4': None,
        '1020_4': None,
        '1021_4': ['allred', 'castro', 'rahgouy'],
        '1022_4': None,
        '1023_4': None,
        '1024_4': ['allred', 'castro', 'rahgouy']
    }
}


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


def text_to_words(text):
    return word_tokenize(text)


def get_jaccard_coefficient(text_1, text_2):
    words_1 = text_to_words(text_1)
    words_2 = text_to_words(text_2)

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


def synonym_substitution(text, jaccard_similarity=0.0, depth=100, n_grams=1, verbose=False):
    original = text
    tokenized_words = text_to_words(text)
    words = []
    for word in range(len(tokenized_words) - n_grams + 1):
        string_of_words = ""
        for index in range(n_grams):
            if index is 0:
                string_of_words += tokenized_words[word]
            else:
                string_of_words += " " + tokenized_words[word + index]
        words.append(string_of_words)

    for word in range(len(words)):
        semantic_similarity = False
        replacement_word = 0
        test_word = words[word]
        experimental_jaccard = get_jaccard_coefficient(original, text)
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
                        or pos_tag([test_word])[0][1] != pos_tag([new_word])[0][1]:
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
                        and pos_tag([test_word])[0][1] == pos_tag([new_word])[0][1]:
                    if verbose:
                        print(test_word)
                        print(new_word)
                        print(experimental_jaccard)
                    text = text.replace(test_word, new_word)
    return text


def remove_parenthetical_phrases(text, keep_proper=False):
    if not keep_proper:
        while text.__contains__('(') or text.__contains__(')'):
            left = text.index('(')
            right = text.index(')')
            if left > 0:
                text = text[:left - 1] + text[right + 1:]
            else:
                text = text[:left] + text[right + 1:]
        output_text = text
    else:
        tags = pos_tag(word_tokenize(text))
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
    tags = pos_tag(word_tokenize(text))
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


def castro_masking(input_directory, verbose=False, output_directory=None, remove_directory=True):
    input_files = listdir(input_directory)

    if output_directory is None:
        author_masking = dirname(dirname(abspath(__file__))) + '/Author Masking Results'
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
        author_masking = dirname(dirname(abspath(__file__))) + '/Author Masking Results'
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
    rmtree(results_directory)


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
        author_masking = dirname(dirname(abspath(__file__))) + '/Author Masking Results'
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
            if search.__contains__(file.split('_')[0]):
                copy(known_directory + "/" + file, input_directory + "/" + search + '/' + file.split('_')[0] + ".txt")

    arguments = ["python3",
                 dirname(dirname(abspath(__file__))) +
                 "/AMT/Rahgouy/author_obf_pan2018/"
                 "obfuscation_script.py", "-i", input_directory, "-o", output_directory,
                 '-lm', pretrained_model, '-t', 'true']

    call(arguments)

    return output_directory


def get_obfuscation_rahgouy(results_directory):
    for file in listdir(results_directory):
        if not isdir(results_directory + '/' + file):
            continue
        copy(results_directory + '/' + file + '/obfuscation.txt', results_directory + '/' + file + '.txt')
        if isdir(results_directory + '/' + file):
            rmtree(results_directory + '/' + file)
        elif isfile(results_directory + '/' + file):
            remove(results_directory + '/' + file)


def lsvm_attribution(input_directory=None, output_directory=None):
    from numpy import array
    from numpy import concatenate
    from numpy import max, min, arange
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.preprocessing import normalize
    from sklearn.svm import LinearSVC

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
            {"unknown_text": texts[i], "author": str(predictions[i]), "score": max(decision_function[i])}) #"score": decision_function[0][i]
    if output_directory is None:
        output_directory = dirname(dirname(abspath(__file__))) + "/Author Attribution Results/LSVM"
    f = open(output_directory + '/answers.json', "w")
    dump({"answers": answers}, f, indent=2)
    f.close()
    return output_directory


def keselj_attribution(input_directory=None, verbose=False, output_directory=None):
    if input_directory is None:
        input_directory = dirname(dirname(abspath(__file__))) + \
                          "/Author Attribution - Input"
    if output_directory is None:
        output_directory = dirname(dirname(abspath(__file__))) + "/Author Attribution Results/Keselj"

    if not isdir(output_directory):
        try:
            mkdir(output_directory)
        except FileExistsError:
            pass
    arguments = ["python3",
                 dirname(dirname(abspath(__file__))) +
                 "/Author Attribution Systems/Keselj/keselj03.py",
                 "-i", input_directory, "-o", output_directory]
    call(arguments)
    return output_directory


def teahan_attribution(input_directory=None, verbose=False, output_directory=None):
    if input_directory is None:
        input_directory = dirname(dirname(abspath(__file__))) + \
                          "/Author Attribution - Input"
        if output_directory is None:
            output_directory = dirname(dirname(abspath(__file__))) + "/Author Attribution Results/Teahan"

    if not isdir(output_directory):
        mkdir(output_directory)

    arguments = ["python3",
                 dirname(dirname(abspath(__file__))) +
                 "/Author Attribution Systems/Teahan/teahan03.py",
                 "-i" + input_directory, "-o" + output_directory]
    call(arguments)
    return output_directory


def koppel_attribution(input_directory=None, output_directory=None):
    if input_directory is None:
        input_directory = dirname(dirname(abspath(__file__))) + \
                          "/Author Attribution - Input"
    if output_directory is None:
        output_directory = dirname(dirname(abspath(__file__))) + "/Author Attribution Results/Koppel"

    if not isdir(output_directory):
        mkdir(output_directory)

    arguments = ["python3",
                 dirname(dirname(abspath(__file__))) +
                 "/Author Attribution Systems/Koppel/koppel11.py",
                 "-i" + input_directory, "-o" + output_directory]
    call(arguments)
    return output_directory


def run_author_attribution(attribution_used, dataset_directory=None, preprocess=True, verbose=False,
                           validation=False, train_files=None, validation_files=None, unknown_files=None,
                           known_files=None, train_directory=None, validation_directory=None, unknown_directory=None,
                           known_directory=None, input_directory=None, truth_file=None, output_directory=None, runs=5):
    answers = {}
    cdaa_preprocess = preprocess

    unknown_mapping = None

    for index in range(len(attribution_used)):
        attribution_used[index] = attribution_used[index].lower()
    if 'lsvm' in attribution_used:
        if preprocess:
            if verbose:
                print('Preprocessing text...')
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
                dataset_directory=dataset_directory, verbose=verbose, validation=True, train_files=train_files,
                validation_files=validation_files, unknown_files=unknown_files, train_directory=train_directory,
                validation_directory=validation_directory, unknown_directory=unknown_directory)
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
            if verbose:
                print('Preprocessing text...')
            train_files, validation_files, unknown_files, unknown_mapping = author_attribution_preprocessing(
                verbose=verbose, train_directory=known_directory, unknown_directory=unknown_directory)
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
            if verbose:
                print('Preprocessing text...')
            train_files, validation_files, unknown_files, unknown_mapping = author_attribution_preprocessing(
                verbose=verbose, train_directory=known_directory, unknown_directory=unknown_directory)
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
            koppel_accuracy = (koppel_accuracy * run + get_answers_accuracy('Koppel', truth_file=truth_file, answers_file=output_directory + '/answers.json')) / (run + 1)
        for answer in potential_answers:
            selected_author = Counter(potential_answers[answer]).most_common(1)[0][0]
            author_score = mean(koppel_scores[answer])
            koppel_answers.append({'unknown_text': answer, 'author': selected_author, 'score': author_score})
        if verbose:
            print('Koppel accuracy: ' + str(koppel_accuracy) + '%\n')
            print()
        answers.update({'Koppel': koppel_answers})

    fitness = fitness_function(answers=answers, truth_file=truth_file)

    return answers, fitness, unknown_mapping


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
                author = 'predicted-author'
                unknown_text = 'unknown-text'
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


def get_answers_accuracy(attribution_name, truth_file=None, answers_file=None):
    answers = get_answers(attribution_name, answers_file)
    ground_truth = get_truth(truth_file)

    correct = 0
    for answer in answers:
        try:
            if ground_truth[answer['unknown_text']] == answer['author']:
                correct += 1
        except KeyError:
            try:
                if ground_truth[answer['unknown-text']] == answer['predicted-author']:
                    correct += 1
            except KeyError:
                raise Exception("Can not find text " + answer['unknown_text'] + '\nGround truth is ' + str(ground_truth))

    return 100 * correct / len(answers)


def get_answers(attribution_name, answers_file=None):
    if answers_file is None:
        answers_file = dirname(dirname(abspath(__file__))) + \
                     "/Author Attribution Results/" + attribution_name + "/answers.json"
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
        truth_file = dirname(dirname(abspath(__file__))) + '/Author Attribution - Input/ground-truth.json'
    if not isfile(truth_file):
        truth_file = dirname(dirname(abspath(__file__))) + '/Author Attribution - Input/problem00001/ground-truth.json'
    with open(truth_file, encoding='utf-8') as json_file:
        ground_truth_data = load(json_file)

    ground_truth = {}
    for file in ground_truth_data['ground-truth']:
        try:
            ground_truth.update({file['unknown-text']: file['true-author']})
        except KeyError:
            ground_truth.update({file['unknown_text']: file['author']})
    return ground_truth


def generate_intial_population(unknown_directory: str, output_directory: str):
    if isdir(output_directory):
        rmtree(output_directory)

    mkdir(output_directory)

    input_files = listdir(unknown_directory)
    for file in input_files:
        mkdir(output_directory + "/" + file.strip('.txt'))
        copy(unknown_directory + "/" + file, output_directory + "/" + file.strip('.txt') + '/' +
             file)
    return output_directory


def generation_driver(file):
    global attribution_system, mask_options, search_directory
    # print('\tWorking on Directory: ' + file)

    for author_masking_system in mask_options:
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
            rahgouy_preprocessing(input_directory=search_directory + '/' + file, known_directory=known_directory)
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

            rahgouy_masking(input_directory=search_directory + '/' + file, known_directory=known_directory,
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

    for index in range(len(attribution_system)):
        attribution_system[index] = attribution_system[index].lower()

    train_files, validation_files, unknown_files, input_directory = author_attribution_preprocessing(
        validation=False, train_directory=known_directory, unknown_directory=search_directory + '/' + file)

    results, fitness_score, unknown_mapping = run_author_attribution(attribution_used=attribution_system, preprocess=False,
                                                    train_directory=dataset_directory + ' - Train',
                                                    validation_directory=dataset_directory + ' - Validation',
                                                    known_directory=known_directory,
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
    global attribution_system
    for index in range(len(attribution_system)):
        attribution_system[index] = attribution_system[index].lower()

    train_files, validation_files, unknown_files, input_directory = author_attribution_preprocessing(
        validation=False, train_directory=known_directory, unknown_directory=search_directory + '/' + file)

    results, fitness_score, unknown_mapping = run_author_attribution(attribution_used=attribution_system, preprocess=False,
                                                    input_directory=input_directory,
                                                    train_directory=dataset_directory + ' - Train',
                                                    validation_directory=dataset_directory + ' - Validation',
                                                    unknown_directory=search_directory + '/' + file,
                                                    known_directory=known_directory, runs=5,
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
        if directory.startswith('.'):
            continue

        candidate = 'candidate' + directory.split('_')[0]

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
    global dataset_directory, search_directory, dataset

    print(search_directory)

    if not isdir(search_directory):
        mkdir(search_directory)

    from socket import gethostname

    if gethostname() == 'Jordan' or not multiprocess:
        processors_used = 1
    else:
        processors_used = ceil(cpu_count() * cpu_fraction)

    pool = Pool(processes=processors_used)

    if dataset_directory is None:
        dataset_directory = 'CASIS Versions/CASIS-25_Dataset'
    if not isdir(dataset_directory):
        dataset_directory = dirname(dirname(abspath(__file__))) + '/' + dataset_directory

    print("Operating With " + str(processors_used) + " CPU(s)")

    if initialize:
        generate_intial_population(unknown_directory=unknown_directory,
                                   output_directory=search_directory)

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


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process input arguments for author attribution preprocessing')

    parser.add_argument('--train', type=str, help='A directory containing the training set in text format', default=None)
    parser.add_argument('--test', type=str, help='A directory containing the test set in text format', default=None)
    parser.add_argument('--output', type=str, help='A complete dataset in the PAN attribution format', default=None)
    parser.add_argument('--preprocess', type=bool, help='Whether or not to create a new input directory using the given train/test sets', default=True)
    parser.add_argument('--multiprocess', type=bool, help='Whether or not to use multiprocessing', default=False)

    parser = parser.parse_args()

    initialize = True
    mask_options = ['castro', 'mihaylova', 'rahgouy']
    attribution_system = ['LSVM']

    message, name = "", ""
    name += '('
    message += "Running using "
    if len(mask_options) == 1:
        message += attribution_system[0].capitalize()
    elif len(mask_options) == 2:
        message += attribution_system[0].capitalize() + ' and ' + attribution_system[1].capitalize()
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
    message += attribution_system[0]
    name += attribution_system[0] + ')'
    message += ' attribution'

    cpu_fraction = 1
    multiprocess = parser.multiprocess
    if parser.output is None:
        search_directory = dirname(dirname(abspath(__file__))) + '/Adversarial Beam Search'
    else:
        search_directory = parser.output

    dataset_directory = 'CASIS Versions/CASIS-25_Dataset'
    if not isdir(dataset_directory):
        dataset_directory = dirname(dirname(abspath(__file__))) + '/' + dataset_directory

    if parser.train is None:
        known_directory = dataset_directory + ' - Known'
    else:
        known_directory = parser.train

    if parser.test is None:
        unknown_directory = dataset_directory + ' - Unknown'
    else:
        unknown_directory = parser.test

    if not initialize:
        dataset = manager.list(listdir(search_directory))
    else:
        dataset = manager.list()
        for file in listdir(unknown_directory):
            dataset.append(file)

    new_dataset = manager.list()
    for file in dataset:
        new_file = file.strip('.txt')
        new_dataset.append(new_file)
    dataset = new_dataset

    beam_size = 20

    print(message)
    evolve_text(initialize=initialize)

    from os import rename
    rename(search_directory,
           search_directory + ' ' + name)

    get_beam_results(data_directory=search_directory + ' ' + name, att_list=attribution_system)

    print('\nDONE')

    # Success!