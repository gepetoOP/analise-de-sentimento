# -*- coding: utf-8 -*-
import re
import sys
import json
import nltk
import time
import random
import string
import codecs
import operator
import collections
import pandas as pd
from io import StringIO
import arff, numpy as np
from nltk.corpus import *
from nltk.util import ngrams
from collections import Counter
from nltk.tokenize import TweetTokenizer
from sklearn.model_selection import KFold
from nltk.metrics import BigramAssocMeasures
from Aelius import Extras, Toqueniza, AnotaCorpus
from nltk.metrics.scores import precision, recall, f_measure



def tokenize(s):
    return tokens_re.findall(s)

def best_word_feats(words, bestwords):
    return dict([(word, True) for word in words if word in bestwords])

def preprocess(s, radica,lowercase=True):
    tokens = tknzr.tokenize(s)
    if lowercase:
        tokens = [token if emoticon_re.search(token) else token.lower() for token in tokens]
    if radical:
        return [stemmer.stem(token) for token in tokens]
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

def information_gain(balanceado):
    pos_word_count_unigrama = condicionalUnigrama['pos'].N()
    pos_word_count_bigrama = condicionalBigrama['pos'].N()
    neg_word_count_unigrama = condicionalUnigrama['neg'].N()
    neg_word_count_bigrama = condicionalBigrama['neg'].N()
    total_word_count_unigrama = pos_word_count_unigrama + neg_word_count_unigrama
    total_word_count_bigrama = pos_word_count_bigrama + neg_word_count_bigrama

    word_scores_unigrama = {}
    word_scores_bigrama = {} 

    for word, freq in frequenciaUnigrama.iteritems():
        pos_score = BigramAssocMeasures.chi_sq(condicionalUnigrama['pos'][word],
            (freq, pos_word_count_unigrama), total_word_count_unigrama)
        neg_score = BigramAssocMeasures.chi_sq(condicionalUnigrama['neg'][word],
            (freq, neg_word_count_unigrama), total_word_count_unigrama)
        word_scores_unigrama[word] = pos_score + neg_score

    for word, freq in frequenciaBigrama.iteritems():
        pos_score = BigramAssocMeasures.chi_sq(condicionalBigrama['pos'][word],
            (freq, pos_word_count_bigrama), total_word_count_bigrama)
        neg_score = BigramAssocMeasures.chi_sq(condicionalBigrama['neg'][word],
            (freq, neg_word_count_bigrama), total_word_count_bigrama)
        word_scores_bigrama[word] = pos_score + neg_score
     
    bestUnigrama = sorted(word_scores_unigrama.iteritems(), key=lambda (w,s): s, reverse=True)[:10000]
    bestBigrama = sorted(word_scores_bigrama.iteritems(), key=lambda (w,s): s, reverse=True)[:10000]

    dicionarioUnigrama = [w for w, s in bestUnigrama]
    dicionarioBigrama = [w for w, s in bestBigrama]

    melhoresUnigramasPositivos = [(best_word_feats(frase, dicionarioUnigrama), 'no') for frase in dicUnigramasPositivos]
    melhoresUnigramasNegativos = [(best_word_feats(frase, dicionarioUnigrama), 'yes') for frase in dicUnigramasNegativos]
    melhoresBigramasPositivos = [(best_word_feats(frase, dicionarioBigrama), 'no') for frase in dicBigramasPositivos]
    melhoresBigramasNegativos = [(best_word_feats(frase, dicionarioBigrama), 'yes') for frase in dicBigramasNegativos]
    return melhoresUnigramasNegativos + melhoresUnigramasPositivos, melhoresBigramasNegativos + melhoresBigramasPositivos
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

datasets = ['OffComBR2']
radical = False
balanceado = False
ig = False
if sys.argv.count('d3'):
    datasets = ['OffComBR3']
if sys.argv.count('todos'):
    datasets = ['OffComBR2','OffComBR3']
if sys.argv.count('radical'):
    radical = True
if sys.argv.count('balanceado'):
    balanceado = True
if sys.argv.count('ig'):
    ig = True

radicais = [True, False]
balanceados = [True, False]
igs = [True, False]
stemmer=nltk.stem.RSLPStemmer()

for base_de_dados in datasets:
    for radical in radicais:
        for balanceado in balanceados:
            for ig in igs:
                dataset = arff.load(base_de_dados+'.arff')
                todosUnigramas = []
                todosBigramas = []
                listaDosUnigramas = []
                listaDosBigramas = []
                negWordsUnigrama = []
                posWordsUnigrama = []
                negWordsBigrama = []
                posWordsBigrama = []
                dicUnigramasPositivos = []
                dicUnigramasNegativos = []
                dicBigramasPositivos = []
                dicBigramasNegativos = []
                for x in dataset:
                    unigramas = [term for term in preprocess(x[1].lower().encode('utf-8'),radical) if term not in stop and not term.startswith(('#', '@'))]
                    bigrama = list(nltk.bigrams(unigramas))
                    bigramas = [' '.join(item) for item in bigrama]
                    if(len(unigramas) > 0):
                        if(x[0] == 'yes'):
                            negWordsUnigrama.extend(unigramas)
                            negWordsBigrama.extend(bigramas)
                            dicUnigramasNegativos.append(unigramas)
                            dicBigramasNegativos.append(bigramas)
                        else:
                            posWordsUnigrama.extend(unigramas)
                            posWordsBigrama.extend(bigramas)
                            dicUnigramasPositivos.append(unigramas)
                            dicBigramasPositivos.append(bigramas)


                        listaDosUnigramas.append({"label": x[0], "dado": unigramas})
                        listaDosBigramas.append({"label": x[0], "dado": bigramas})

                frequenciaUnigrama = nltk.FreqDist()
                condicionalUnigrama = nltk.ConditionalFreqDist()
                dicionarioUnigrama = []

                frequenciaBigrama = nltk.FreqDist()
                condicionalBigrama = nltk.ConditionalFreqDist()
                dicionarioBigrama = []

                for word in posWordsUnigrama:
                    frequenciaUnigrama[word.lower()] += 1
                    condicionalUnigrama['pos'][word.lower()] += 1

                for word in posWordsBigrama:
                    frequenciaBigrama[word.lower()] += 1
                    condicionalBigrama['pos'][word.lower()] += 1
                 
                for word in negWordsUnigrama:
                    frequenciaUnigrama[word.lower()] += 1
                    condicionalUnigrama['neg'][word.lower()] += 1

                for word in negWordsBigrama:
                    frequenciaBigrama[word.lower()] += 1
                    condicionalBigrama['neg'][word.lower()] += 1

                if len(dicUnigramasPositivos) > len(dicUnigramasNegativos):
                    menor_tamanho = len(dicUnigramasNegativos)
                else:
                    menor_tamanho = len(dicUnigramasPositivos)

                if ig:
                    featuresetUnigrama,featuresetBigrama = information_gain(balanceado)
                else:
                    todosBigramas = posWordsBigrama + negWordsBigrama
                    todosUnigramas = negWordsUnigrama + posWordsUnigrama

                    if(balanceado):
                        featuresetUnigrama = [(extraiFeatures(frase,todosUnigramas),'no') for frase in dicUnigramasPositivos] + [(extraiFeatures(frase,todosUnigramas),'yes') for frase in dicUnigramasNegativos]
                        featuresetBigrama = [(extraiFeatures(frase,todosBigramas),'no') for frase in dicBigramasPositivos] + [(extraiFeatures(frase,todosBigramas),'yes') for frase in dicBigramasNegativos]
                    else:
                        featuresetUnigrama = [(extraiFeatures(frase,todosUnigramas),'no') for frase in dicUnigramasPositivos] + [(extraiFeatures(frase,todosUnigramas),'yes') for frase in dicUnigramasNegativos]
                        featuresetBigrama = [(extraiFeatures(frase,todosBigramas),'no') for frase in dicBigramasPositivos] + [(extraiFeatures(frase,todosBigramas),'yes') for frase in dicBigramasNegativos]
                resultados = [
                    {
                        'nome': 'Unigrama',
                        'arquivo': 'ResultadosUnigrama.txt',
                        'featureset': featuresetUnigrama
                    },
                    {
                        'nome': 'Bigrama',
                        'arquivo': 'ResultadosBigrama.txt',
                        'featureset': featuresetBigrama
                    }
                ]

                for resultado in resultados:
            # testeP = ["te", "amo", "muito"]
            # testeN = ["velho", "asqueroso"]
                    featureset = resultado['featureset']
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

                    arquivo = open(resultado['arquivo'], 'a')
                    arquivo.write('Balanceado: {}'.format(balanceado))
                    arquivo.write('\nDataset: {}'.format(base_de_dados))
                    arquivo.write('\nInformation Gain: {}'.format(ig))
                    arquivo.write('\nRadical: {}'.format(radical))
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
                    arquivo.write('\n\n\n')
                    arquivo.close()


# texto = []
# positivos = []
# negativos = []

# featureset = [(extraiFeatures(tweet['dado'],dicionarioBigrama), tweet['label']) for tweet in bigramaTotal]
# random.shuffle(featureset)
# tamanho = len(featureset)
# kf = KFold(n_splits=10)
# sum = 0
# precisionPositivoMedia = 0;
# precisionNegativoMedia = 0;
# recallPositivoMedia = 0;
# recallNegativoMedia = 0;
# fmeasurePositivoMedia = 0;
# fmeasureNegativoMedia = 0;
# for train, test in kf.split(featureset):
#     refsets = collections.defaultdict(set)
#     testsets = collections.defaultdict(set)
#     train_data = np.array(featureset)[train]
#     test_data = np.array(featureset)[test]
#     classifier = nltk.NaiveBayesClassifier.train(train_data)
#     sum += nltk.classify.accuracy(classifier, test_data)
#     for i, (feats, label) in enumerate(test_data):
#         refsets[label].add(i)
#         observed = classifier.classify(feats)
#         testsets[observed].add(i)
#     precisionPositivoMedia += precision(refsets['yes'], testsets['yes'])
#     precisionNegativoMedia += precision(refsets['no'], testsets['no'])
#     recallPositivoMedia += recall(refsets['yes'], testsets['yes'])
#     recallNegativoMedia += recall(refsets['no'], testsets['no'])
#     fmeasurePositivoMedia += f_measure(refsets['yes'], testsets['yes'])
#     fmeasureNegativoMedia += f_measure(refsets['no'], testsets['no'])
#     wPositivoMedia = (1 + (0.05*0.05)) * (precisionPositivoMedia*recallPositivoMedia)/((precisionPositivoMedia*(0.05*0.05))+recallPositivoMedia)
#     wNegativoMedia = (1 + (0.05*0.05)) * (precisionNegativoMedia*recallNegativoMedia)/((precisionNegativoMedia*(0.05*0.05))+recallNegativoMedia)
# average = sum/10
# precisionPositivoMedia = precisionPositivoMedia/10
# precisionNegativoMedia = precisionNegativoMedia/10
# recallPositivoMedia = recallPositivoMedia/10
# recallNegativoMedia = recallNegativoMedia/10
# fmeasurePositivoMedia = 2 * (precisionPositivoMedia*recallPositivoMedia)/(precisionPositivoMedia+recallPositivoMedia)
# fmeasureNegativoMedia = 2 * (precisionNegativoMedia*recallNegativoMedia)/(precisionNegativoMedia+recallNegativoMedia)
# wPositivoMedia = wPositivoMedia/10
# wNegativoMedia = wNegativoMedia/10

# mostInf = classifier.show_most_informative_features(10)

# arquivo = open('resultadosBigram.txt', 'a')
# arquivo.write('\nPositivo Precision: ')
# arquivo.write(str(precisionPositivoMedia))
# arquivo.write('\nPositivo Recall: ')
# arquivo.write(str(recallPositivoMedia))
# arquivo.write('\nPositivo F-Measure: ')
# arquivo.write(str(fmeasurePositivoMedia))
# arquivo.write('\nPositivo W - F-Measure: ')
# arquivo.write(str(wPositivoMedia))
# arquivo.write('\nNegativo Precision: ')
# arquivo.write(str(precisionNegativoMedia))
# arquivo.write('\nNegativo Recall: ')
# arquivo.write(str(recallNegativoMedia))
# arquivo.write('\nNegativo F-measure: ')
# arquivo.write(str(fmeasureNegativoMedia))
# arquivo.write('\nNegativo W - F-Measure: ')
# arquivo.write(str(wNegativoMedia))
# arquivo.write('\nMedia Unigram: ')
# arquivo.write(str(average))
# arquivo.write('\nTamanho do Arquivo:')
# arquivo.write(str(tamanho))
# arquivo.close()
