# -*- coding: utf-8 -*-
import re
import sys
import json
import nltk
import time
import math
import random
import string
import codecs
import goslate
import operator
import collections
import pandas as pd
from io import StringIO
import arff, numpy as np
import matplotlib.pyplot as plt
from nltk.corpus import *
import concurrent.futures
from nltk.util import ngrams
from collections import Counter
from googletrans import Translator
from nltk.tokenize import TweetTokenizer
from sklearn.model_selection import KFold
from nltk.metrics import BigramAssocMeasures
from Aelius import Extras, Toqueniza, AnotaCorpus
from nltk.metrics.scores import precision, recall, f_measure
# from google.cloud import translate
# from translate import Translator




def tokenize(s):
    return tokens_re.findall(s)

def best_word_feats(words, bestwords):
    return dict([(word, True) for word in words if word in bestwords])

def preprocess(s, radical,lowercase=True):
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

def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2., height,
                str(int(round(height,2)*100))+'%',
                ha='center', va='bottom')

def incrementa_dataset(dataset):
    epFile = open("espanholParaPortuguesOFF2.txt", "r")
    inFile = open("inglesParaPortuguesOFF2.txt", "r")
    espanhol, ingles = [], []
    texto = epFile.readlines()
    texto2 = inFile.readlines()
    for frase in texto:
        espanhol.append(frase)
    for frase in texto2:
        ingles.append((frase))
    epFile.close()
    inFile.close()
    frasesComInglesEspanhol = []
    frases = []
    classes = []
    conjuntoEspanhol = []
    conjuntoNormal = []
    conjuntoIngles = []
    for x in dataset:
        frases.append(x[1])
        classes.append(x[0])
    for i in range(len(frases)):
        if(espanhol[i] == '\n' or ingles[i] == '\n'):
            continue
        else:
            conjuntoEspanhol.append((classes[i],espanhol[i]))
            conjuntoIngles.append((classes[i],ingles[i]))
            conjuntoNormal.append((classes[i],frases[i]))
    frasesComInglesEspanhol = conjuntoEspanhol + conjuntoNormal + conjuntoIngles
    data = [(classe, frase) for (classe, frase) in frasesComInglesEspanhol]
    # print(frasesComInglesEspanhol)
    print arff.dumps(frasesComInglesEspanhol)

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
    datasets = ['OffComBR2','OffComBR3','portuguesInglesEspanholBR2','portuguesInglesEspanholBR3']
if sys.argv.count('radical'):
    radical = True
if sys.argv.count('balanceado'):
    balanceado = True
if sys.argv.count('pl'):
    pl = True
if sys.argv.count('ig'):
    ig = True


# radicais = [False]
# balanceados = [False]
# igs = [False]
radicais = [False,True]
balanceados = [False, True]
igs = [False, True]
stemmer=nltk.stem.RSLPStemmer()
# dataset = arff.load('OffComBR2.arff')
# incrementa_dataset(dataset)

for base_de_dados in datasets:
    resultadosParaPlotar = []
    nomesDosResultadosParaPlotar = []
    for radical in radicais:
        for balanceado in balanceados:
            for ig in igs:
                temIg = '+IG' if ig == True else ''
                temBalanceado = '+BA' if balanceado == True else ''
                temRadical = '+RA' if radical == True else ''
                nomeDoResultadoUnigrama = '1G'+temIg+temBalanceado+temRadical
                nomeDoResultadoBigrama = '2G'+temIg+temBalanceado+temRadical
                nomesDosResultadosParaPlotar.append(nomeDoResultadoUnigrama)
                nomesDosResultadosParaPlotar.append(nomeDoResultadoBigrama)
                dataset = arff.load(base_de_dados+'.arff')
                todosUnigramas = []
                todosBigramas = []
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
                    featureset = resultado['featureset']
                    random.shuffle(featureset)
                    tamanho = len(featureset)
                    kf = KFold(n_splits=10)
                    soma = 0
                    precisionPositivoMedia = 0;
                    precisionNegativoMedia = 0;
                    recallPositivoMedia = 0;
                    recallNegativoMedia = 0;
                    fmeasurePositivoMedia = 0;
                    fmeasureNegativoMedia = 0;
                    wPositivoMedia = 0;
                    wNegativoMedia = 0;
                    for train, test in kf.split(featureset):
                        refsets = collections.defaultdict(set)
                        testsets = collections.defaultdict(set)
                        train_data = np.array(featureset)[train]
                        test_data = np.array(featureset)[test]
                        classifier = nltk.NaiveBayesClassifier.train(train_data)
                        soma += nltk.classify.accuracy(classifier, test_data)
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
                        # wPositivoMedia += (1 + (0.05*0.05)) * (precisionPositivoMedia*recallPositivoMedia)/((precisionPositivoMedia*(0.05*0.05))+recallPositivoMedia)
                        # wNegativoMedia += (1 + (0.05*0.05)) * (precisionNegativoMedia*recallNegativoMedia)/((precisionNegativoMedia*(0.05*0.05))+recallNegativoMedia)
                    average = soma/10
                    precisionPositivoMedia = precisionPositivoMedia/10
                    precisionNegativoMedia = precisionNegativoMedia/10
                    recallPositivoMedia = recallPositivoMedia/10
                    recallNegativoMedia = recallNegativoMedia/10
                    fmeasurePositivoMedia = 2 * (precisionPositivoMedia*recallPositivoMedia)/(precisionPositivoMedia+recallPositivoMedia)
                    fmeasureNegativoMedia = 2 * (precisionNegativoMedia*recallNegativoMedia)/(precisionNegativoMedia+recallNegativoMedia)
                    # wPositivoMedia = wPositivoMedia/10
                    # wNegativoMedia = wNegativoMedia/10

                    # mostInf = classifier.show_most_informative_features(10)
                    # print(fmeasurePositivoMedia,fmeasureNegativoMedia,mediaFmeasure)
                    resultadosParaPlotar.append((fmeasurePositivoMedia+fmeasureNegativoMedia)/2)
                    # arquivo = open(resultado['arquivo'], 'a')
                    # arquivo.write('Balanceado: {}'.format(balanceado))
                    # arquivo.write('\nDataset: {}'.format(base_de_dados))
                    # arquivo.write('\nInformation Gain: {}'.format(ig))
                    # arquivo.write('\nRadical: {}'.format(radical))
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
                    # arquivo.write('\n\n\n')
                    # arquivo.close()
    fig, ax = plt.subplots()
    fig.subplots_adjust(bottom=0.45)
    barra = ax.bar(nomesDosResultadosParaPlotar,resultadosParaPlotar, width = 0.8, color="red")
    ax.set_yticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
    ax.set_xticklabels(nomesDosResultadosParaPlotar,rotation='vertical')
    ax.set_ylabel('F-Measure')
    ax.set_xlabel(base_de_dados)
    autolabel(barra)
    plt.savefig(base_de_dados+'.png')
    # plt.show()
