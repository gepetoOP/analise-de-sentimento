# -*- coding: utf-8 -*-
import random
import string
import operator
import json
import re
import nltk
import time
import codecs
import collections
import pandas as pd
from io import StringIO
import arff, numpy as np
from nltk.corpus import *
from nltk.util import ngrams
from collections import Counter
from nltk.tokenize import TweetTokenizer
from sklearn.model_selection import KFold
from Aelius import Extras, Toqueniza, AnotaCorpus
from nltk.metrics.scores import precision, recall, f_measure

inicio = time.time()

punctuation = list(string.punctuation)
stop = stopwords.words('portuguese') + punctuation + ['rt', 'via']

emoticons_str = r"""
    (?:
        [:=;] # Eyes
        [oO\-]? # Nose (optional)
        [D\)\]\(\]/\\OpP] # Mouth
    )"""

regex_str = [
    emoticons_str,
    r'<[^>]+>',  # HTML tags
    r'(?:@[\w_]+)',  # @-mentions
    r"(?:\#+[\w_]+[\w\'_\-]*[\w_]+)",  # hash-tags
    r'http[s]?://(?:[a-z]|[0-9]|[$-_@.&amp;+]|[!*\(\),]|(?:%[0-9a-f][0-9a-f]))+',  # URLs

    r'(?:(?:\d+,?)+(?:\.?\d+)?)',  # numbers
    r"(?:[a-z][a-z'\-_]+[a-z])",  # words with - and '
    r'(?:[\w_]+)',  # other words
    r'(?:\S)'  # anything else
]

tokens_re = re.compile(r'(' + '|'.join(regex_str) + ')', re.VERBOSE | re.IGNORECASE)
emoticon_re = re.compile(r'^' + emoticons_str + '$', re.VERBOSE | re.IGNORECASE)
tknzr = TweetTokenizer()


def tokenize(s):
    return tokens_re.findall(s)


def preprocess(s, lowercase=True):
    tokens = tknzr.tokenize(s)
    if lowercase:
        tokens = [token if emoticon_re.search(token) else token.lower() for token in tokens]
    return tokens

def etiqueta(tokens):
    return AnotaCorpus.anota_sentencas([tokens], maxent, "mxpost")


def extraiFeatures(frase,dicionario):
    features = {}
    for word in dicionario:
        try:
            if word in frase:
                features["{}".format(word.encode('utf-8'))] = True
        except Exception as inst:
            print(inst)
            continue
    return features

dataset = arff.load('OffComBR2.arff')
stemmer=nltk.stem.RSLPStemmer()
# uni_file = open('uniPreProcessado.txt', 'w')
# big_file = open('bigPreProcessado.txt', 'w')
todosUnigramas = []
todosBigramas = []
listaDosUnigramas = []
listaDosBigramas = []
posUnigramas = []
negUnigramas = []
posBigramas = []
negBigramas = []
lun = []
lup = []
lbp = []
lbn = []

for x in dataset:
    unigramas = [term for term in preprocess(x[1].lower().encode('utf-8')) if term not in stop and not term.startswith(('#', '@'))]
    bigrama = list(nltk.bigrams(unigramas))
    bigramas = [' '.join(item) for item in bigrama]
    if(len(unigramas) > 0):
        if(x[0] == 'yes'):
            lun.append({"label": x[0], "dado": unigramas})
            lbn.append({"label": x[0], "dado": bigramas})
            negUnigramas.extend(unigramas)
            negBigramas.extend(bigramas)
        else:
            posUnigramas.extend(unigramas)
            posBigramas.extend(bigramas)
            lup.append({"label": x[0], "dado": unigramas})
            lbp.append({"label": x[0], "dado": bigramas})
        todosUnigramas.extend(unigramas)
        # print >> uni_file,{"label": x[0], "dado": unigramas}
        todosBigramas.extend(bigramas)
        # print >> big_file,{"label": x[0], "dado": bigramas}

        listaDosUnigramas.append({"label": x[0], "dado": unigramas})
        listaDosBigramas.append({"label": x[0], "dado": bigramas})


# print(posUnigramas)
# print(len(posUnigramas), len(negUnigramas))
totalDeUnigramas = posUnigramas[:len(negUnigramas)] + negUnigramas
totalDeBigramas = posBigramas[:len(negBigramas)] + negBigramas
frequenciaUnigramaPos = nltk.FreqDist(posUnigramas)
frequenciaUnigramaNeg = nltk.FreqDist(negUnigramas)
dicionarioUnigrama = []
# print(len(posBigramas),len(negBigramas))
frequenciaBigramaPos = nltk.FreqDist(posBigramas)
frequenciaBigramaNeg = nltk.FreqDist(negBigramas)
dicionarioBigrama = []

for frequency in frequenciaUnigramaPos.most_common():
    dicionarioUnigrama.append(frequency[0])
for frequency in frequenciaUnigramaNeg.most_common():
    dicionarioUnigrama.append(frequency[0])
for frequency in frequenciaBigramaPos.most_common():
    dicionarioBigrama.append(frequency[0])
for frequency in frequenciaBigramaNeg.most_common():
    dicionarioBigrama.append(frequency[0])

unigramaTotal = lun + lup[:len(lun)]
bigramaTotal = lbn + lbp[:len(lbn)]
# print bigramaTotal

# texto = []
# positivos = []
# negativos = []

featureset = [(extraiFeatures(tweet['dado'],dicionarioUnigrama), tweet['label']) for tweet in unigramaTotal]
# # testeP = ["te", "amo", "muito"]
# # testeN = ["velho", "asqueroso"]
random.shuffle(featureset)
tamanho = len(featureset)
kf = KFold(n_splits=10)
sum = 0
precisionPositivoMedia = 0;
precisionNegativoMedia = 0;
recallPositivoMedia = 0;
recallNegativoMedia = 0;
fmeasurePositivoMedia = 0;
fmeasureNegativoMedia = 0;
for train, test in kf.split(featureset):
    refsets = collections.defaultdict(set)
    testsets = collections.defaultdict(set)
    train_data = np.array(featureset)[train]
    test_data = np.array(featureset)[test]
    classifier = nltk.NaiveBayesClassifier.train(train_data)
    sum += nltk.classify.accuracy(classifier, test_data)
    for i, (feats, label) in enumerate(test_data):
        refsets[label].add(i)
        observed = classifier.classify(feats)
        testsets[observed].add(i)
    precisionPositivoMedia += precision(refsets['yes'], testsets['yes'])
    precisionNegativoMedia += precision(refsets['no'], testsets['no'])
    recallPositivoMedia += recall(refsets['yes'], testsets['yes'])
    recallNegativoMedia += recall(refsets['no'], testsets['no'])
    fmeasurePositivoMedia += f_measure(refsets['yes'], testsets['yes'])
    fmeasureNegativoMedia += f_measure(refsets['no'], testsets['no'])
    wPositivoMedia = (1 + (0.05*0.05)) * (precisionPositivoMedia*recallPositivoMedia)/((precisionPositivoMedia*(0.05*0.05))+recallPositivoMedia)
    wNegativoMedia = (1 + (0.05*0.05)) * (precisionNegativoMedia*recallNegativoMedia)/((precisionNegativoMedia*(0.05*0.05))+recallNegativoMedia)
average = sum/10
precisionPositivoMedia = precisionPositivoMedia/10
precisionNegativoMedia = precisionNegativoMedia/10
recallPositivoMedia = recallPositivoMedia/10
recallNegativoMedia = recallNegativoMedia/10
fmeasurePositivoMedia = 2 * (precisionPositivoMedia*recallPositivoMedia)/(precisionPositivoMedia+recallPositivoMedia)
fmeasureNegativoMedia = 2 * (precisionNegativoMedia*recallNegativoMedia)/(precisionNegativoMedia+recallNegativoMedia)
wPositivoMedia = wPositivoMedia/10
wNegativoMedia = wNegativoMedia/10


mostInf = classifier.show_most_informative_features(10)

arquivo = open('resultadosUnigram.txt', 'a')
arquivo.write('\nPositivo Precision: ')
arquivo.write(str(precisionPositivoMedia))
arquivo.write('\nPositivo Recall: ')
arquivo.write(str(recallPositivoMedia))
arquivo.write('\nPositivo F-Measure: ')
arquivo.write(str(fmeasurePositivoMedia))
arquivo.write('\nPositivo W - F-Measure: ')
arquivo.write(str(wPositivoMedia))
arquivo.write('\nNegativo Precision: ')
arquivo.write(str(precisionNegativoMedia))
arquivo.write('\nNegativo Recall: ')
arquivo.write(str(recallNegativoMedia))
arquivo.write('\nNegativo F-measure: ')
arquivo.write(str(fmeasureNegativoMedia))
arquivo.write('\nNegativo W - F-Measure: ')
arquivo.write(str(wNegativoMedia))
arquivo.write('\nMedia Unigram: ')
arquivo.write(str(average))
arquivo.write('\nTamanho do Arquivo:')
arquivo.write(str(tamanho))
arquivo.close()


texto = []
positivos = []
negativos = []

featureset = [(extraiFeatures(tweet['dado'],dicionarioBigrama), tweet['label']) for tweet in bigramaTotal]
random.shuffle(featureset)
tamanho = len(featureset)
kf = KFold(n_splits=10)
sum = 0
precisionPositivoMedia = 0;
precisionNegativoMedia = 0;
recallPositivoMedia = 0;
recallNegativoMedia = 0;
fmeasurePositivoMedia = 0;
fmeasureNegativoMedia = 0;
for train, test in kf.split(featureset):
    refsets = collections.defaultdict(set)
    testsets = collections.defaultdict(set)
    train_data = np.array(featureset)[train]
    test_data = np.array(featureset)[test]
    classifier = nltk.NaiveBayesClassifier.train(train_data)
    sum += nltk.classify.accuracy(classifier, test_data)
    for i, (feats, label) in enumerate(test_data):
        refsets[label].add(i)
        observed = classifier.classify(feats)
        testsets[observed].add(i)
    precisionPositivoMedia += precision(refsets['yes'], testsets['yes'])
    precisionNegativoMedia += precision(refsets['no'], testsets['no'])
    recallPositivoMedia += recall(refsets['yes'], testsets['yes'])
    recallNegativoMedia += recall(refsets['no'], testsets['no'])
    fmeasurePositivoMedia += f_measure(refsets['yes'], testsets['yes'])
    fmeasureNegativoMedia += f_measure(refsets['no'], testsets['no'])
    wPositivoMedia = (1 + (0.05*0.05)) * (precisionPositivoMedia*recallPositivoMedia)/((precisionPositivoMedia*(0.05*0.05))+recallPositivoMedia)
    wNegativoMedia = (1 + (0.05*0.05)) * (precisionNegativoMedia*recallNegativoMedia)/((precisionNegativoMedia*(0.05*0.05))+recallNegativoMedia)
average = sum/10
precisionPositivoMedia = precisionPositivoMedia/10
precisionNegativoMedia = precisionNegativoMedia/10
recallPositivoMedia = recallPositivoMedia/10
recallNegativoMedia = recallNegativoMedia/10
fmeasurePositivoMedia = 2 * (precisionPositivoMedia*recallPositivoMedia)/(precisionPositivoMedia+recallPositivoMedia)
fmeasureNegativoMedia = 2 * (precisionNegativoMedia*recallNegativoMedia)/(precisionNegativoMedia+recallNegativoMedia)
wPositivoMedia = wPositivoMedia/10
wNegativoMedia = wNegativoMedia/10

mostInf = classifier.show_most_informative_features(10)

arquivo = open('resultadosBigram.txt', 'a')
arquivo.write('\nPositivo Precision: ')
arquivo.write(str(precisionPositivoMedia))
arquivo.write('\nPositivo Recall: ')
arquivo.write(str(recallPositivoMedia))
arquivo.write('\nPositivo F-Measure: ')
arquivo.write(str(fmeasurePositivoMedia))
arquivo.write('\nPositivo W - F-Measure: ')
arquivo.write(str(wPositivoMedia))
arquivo.write('\nNegativo Precision: ')
arquivo.write(str(precisionNegativoMedia))
arquivo.write('\nNegativo Recall: ')
arquivo.write(str(recallNegativoMedia))
arquivo.write('\nNegativo F-measure: ')
arquivo.write(str(fmeasureNegativoMedia))
arquivo.write('\nNegativo W - F-Measure: ')
arquivo.write(str(wNegativoMedia))
arquivo.write('\nMedia Unigram: ')
arquivo.write(str(average))
arquivo.write('\nTamanho do Arquivo:')
arquivo.write(str(tamanho))
arquivo.close()
