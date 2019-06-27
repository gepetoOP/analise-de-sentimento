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
from unidecode import unidecode
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
from unicodedata import normalize
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
# from google.cloud import translate
# from translate import Translator
posWordsUnigramaTeste = []
posWordsBigramaTeste = []
dicUnigramasPositivosTeste = []
dicBigramasPositivosTeste = []
negWordsUnigramaTeste = []
negWordsBigramaTeste = []
dicUnigramasNegativosTeste = []
dicBigramasNegativosTeste = []
resultadosParaPlotar = []
nomesDosResultadosParaPlotar = []

def remover_acentos(txt, codif='utf-8'):
    return normalize('NFKD', txt.decode(codif)).encode('ASCII', 'ignore')

def tokenize(s):
    return tokens_re.findall(s)

def best_word_feats(words, bestwords):
    return dict([(word, True) for word in words if word in bestwords])

def preprocess(s, radical,lowercase=True):
    frase = remove_re.sub('',s)
    tokens = tokenize(frase)
    if lowercase:
        return [token.lower() for token in tokens]
    if radical:
        return [stemmer.stem(token.lower()) for token in tokens]
    # return tokens

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

def unificaNegativos(arquivo):
    f = open('negativosAlemaoOriginal.txt','w')
    alemaoFile = open('negativosAlemaoPortuguesOFF2.txt','r')
    espanholFile = open('negativosEspanholPortuguesOFF2.txt','r')
    inglesFile = open('negativosInglesPortuguesOFF2.txt','r')
    alemao = []
    espanhol = []
    ingles = []
    original = []
    for frases in arquivo:
        original.append(frases)
    for frases in alemaoFile:
        if(frases != '\n'):
            alemao.append(frases)
    for frases in espanholFile:
        if(frases != '\n'):
            espanhol.append(frases)
    for frases in inglesFile:
        if(frases != '\n'):
            ingles.append(frases)
    original.extend(alemao)
    random.shuffle(original)
    for frases in original:
        print >> f, frases[:-1]

def data_augmentation(dataset):
    with open('indicesNegativosOFF'+dataset[-1]+'.txt','r') as f:
        data = f.readlines()
        ind = [-1]*1250
        for indice,indiceNegativo in enumerate(data):
            ind[int(indiceNegativo[:-1])] = indice
    return ind

def procura_indiceBR2_BR3():
    dataset2 = arff.load('OffComBR2.arff')
    dataset3 = arff.load('OffComBR3.arff')
    palavras3 = []
    indices3 = [] 
    indices2 = []
    cont = 0
    palavras2 = []
    comum = []
    for x in dataset3:
        if(x[0] == 'yes'):
            palavras3.append(x[1])
            indices3.append(cont)
        cont += 1
    cont = 0
    for x in dataset2:
        if(x[0] == 'yes'):
            palavras2.append(x[1])
            indices2.append(cont)
        cont += 1
    for indice,palavra in enumerate(palavras2):
        for pl in palavras3:
            if(pl == palavra):
                comum.append(indice)
    teste = set(comum)
    comum = list(teste)
    dados = []
    f = open('negativosOFF3.txt','w')
    data = palavras3
    for indice,d in enumerate(data):
        print >> f, d

def unique(lista):
    return list(dict.fromkeys(lista))

def aumenta_dados(train_index,ind_balanceado,conjunto):
    frasesParaAumentar = []
    if(base_de_dados == 'pt-de-en-in-BR3'):
        for index in train_index:
            if(ind_balanceado[index] != -1 and len(frasesNegativas[ind_balanceado[index]]) > 1):
                frasesParaAumentar.append(frasesNegativas[ind_balanceado[index]])
                frasesParaAumentar.append(frasesNegativas[ind_balanceado[index]+202])
                frasesParaAumentar.append(frasesNegativas[ind_balanceado[index]+404])
    else:
        for index in train_index:
            if(ind_balanceado[index] != -1 and len(frasesNegativas[ind_balanceado[index]]) > 1):
                frasesParaAumentar.append(frasesNegativas[ind_balanceado[index]])
    random.shuffle(frasesNegativas)
    unigramas = []
    bigramas = []
    listaUnigrama = []
    listaBigrama = []
    for frase in frasesParaAumentar:
        unigramas = [term for term in preprocess(remover_acentos(frase).lower().encode('utf-8'),False) if term not in stop and len(term) > 1]
        bigrama = list(nltk.bigrams(unigramas))
        bigramas = [' '.join(item) for item in bigrama]
        listaUnigrama.append(unigramas)
        listaBigrama.append(bigramas)
    if(conjunto == 'Bigrama'):
        lista = [(extraiFeatures(frase,todosBigramas),'yes') for frase in listaBigrama]
    if(conjunto == 'Unigrama'):
        lista = [(extraiFeatures(frase,todosUnigramas),'yes') for frase in listaUnigrama]
    return lista

def incrementa_dataset(dataset):
    alFile = open("negativosAlemaoPortuguesOFF3.txt","r")
    epFile = open("negativosEspanholPortuguesOFF3.txt", "r")
    inFile = open("negativosInglesPortuguesOFF3.txt", "r")
    espanhol, ingles, alemao = [], [], []
    texto = inFile.readlines()
    texto2 = epFile.readlines()
    texto3 = alFile.readlines()
    data = []
    frases = []
    classes = []
    conjuntoEspanhol = []
    conjuntoNormal = []
    conjuntoIngles = []
    conjuntoAlemao = []
    for x in dataset:
        conjuntoNormal.append((x[0],x[1]))
        classes.append(x[0])
    for frase in texto:
        conjuntoNormal.append(('yes',remover_acentos(frase[:-1])))
    for frase in texto2:
        conjuntoNormal.append(('yes',remover_acentos(frase[:-1])))
    for frase in texto3:
        conjuntoNormal.append(('yes',remover_acentos(frase[:-1])))
    epFile.close()
    inFile.close()
    alFile.close()
    # for i in range(len(frases)):
        # if(espanhol[i] == '\n' or ingles[i] == '\n'):
            # continue
        # else:
            # conjuntoEspanhol.append((classes[i],espanhol[i]))
            # conjuntoAlemao.append(('yes',alemao[i]))
            # conjuntoIngles.append((classes[i],ingles[i]))
        # conjuntoNormal.append((classes[i],frases[i]))
    # conjuntoNormal.extend(conjuntoAlemao) 

    # data = [(classe, frase) for (classe, frase) in frasesComInglesEspanhol]
    # print(frasesComInglesEspanhol)
    print arff.dumps(conjuntoNormal)

def escreve_negativos(dataset):
    with open('indicesNegativosOFF3.txt','w') as f:
        for indice,x in enumerate(dataset):
            if(x[0] == 'yes'):
                f.write(str(indice)+'\n')

def features_com_base_kaggle(radical,balanceado,ig):w
    dataset = pd.read_csv('Tweets_Mg.csv',encoding='utf-8')
    print dataset.Classificacao == 'Negativo'
    positivo = set(dataset[dataset.Classificacao == 'Positivo']['Text'].values)
    negativo = set(dataset[dataset.Classificacao == 'Negativo']['Text'].values)
    dicBigramasPositivosTeste = set()
    dicUnigramasPositivosTeste = set()
    dicBigramasNegativosTeste = set()
    dicUnigramasNegativosTeste = set()
    posWordsUnigramaTeste = []
    posWordsBigramaTeste = []
    negWordsUnigramaTeste = []
    negWordsBigramaTeste = []
    resultadosParaPlotar = set()
    nomesDosResultadosParaPlotar = set()

    for frase in positivo:
        unigramas = [term for term in preprocess(remover_acentos(frase.lower().encode('utf-8')),radical) if term not in stop]
        bigrama = list(nltk.bigrams(unigramas))
        bigramas = [' '.join(item) for item in bigrama]
        if(len(unigramas) > 0):
            posWordsUnigramaTeste.extend(unigramas)
            posWordsBigramaTeste.extend(bigramas)
            dicUnigramasPositivosTeste.add(tuple(unigramas))
            dicBigramasPositivosTeste.add(tuple(bigramas))
    for frase in negativo:
        unigramas = [term for term in preprocess(remover_acentos(frase.lower().encode('utf-8')),radical) if term not in stop]
        bigrama = list(nltk.bigrams(unigramas))
        bigramas = [' '.join(item) for item in bigrama]
        if(len(unigramas) > 0):
            negWordsUnigramaTeste.extend(unigramas)
            negWordsBigramaTeste.extend(bigramas)
            dicUnigramasNegativosTeste.add(tuple(unigramas))
            dicBigramasNegativosTeste.add(tuple(bigramas))
    # print dicBigramasNegativosTeste
    negWordsUnigramaTeste = unique(negWordsUnigramaTeste)
    negWordsBigramaTeste = unique(negWordsBigramaTeste)
    posWordsUnigramaTeste = unique(posWordsUnigramaTeste)
    posWordsBigramaTeste = unique(posWordsBigramaTeste)
    if len(negWordsUnigramaTeste) > len(posWordsUnigramaTeste):
        tam = len(posWordsUnigramaTeste)
    else:
        tam = len(negWordsUnigramaTeste)
    if(balanceado):
        todosBigramasTeste = posWordsBigramaTeste[:tam] + negWordsBigramaTeste[:tam]
        todosUnigramasTeste = posWordsUnigramaTeste[:tam] + negWordsUnigramaTeste[:tam]
    else:
        todosBigramasTeste = posWordsBigramaTeste + negWordsBigramaTeste
        todosUnigramasTeste = posWordsUnigramaTeste + negWordsUnigramaTeste
    featuresetUnigramaTeste = [(extraiFeatures(frase,todosUnigramasTeste),'no') for frase in dicUnigramasPositivosTeste] + [(extraiFeatures(frase,todosUnigramasTeste),'yes') for frase in dicUnigramasNegativosTeste]
    featuresetBigramaTeste = [(extraiFeatures(frase,todosBigramasTeste),'no') for frase in dicBigramasPositivosTeste] + [(extraiFeatures(frase,todosBigramasTeste),'yes') for frase in dicBigramasNegativosTeste]
    if ig:
        frequenciaUnigrama = nltk.FreqDist()
        condicionalUnigrama = nltk.ConditionalFreqDist()
        dicionarioUnigrama = []

        frequenciaBigrama = nltk.FreqDist()
        condicionalBigrama = nltk.ConditionalFreqDist()
        dicionarioBigrama = []

        for word in posWordsUnigramaTeste:
            frequenciaUnigrama[word.lower()] += 1
            condicionalUnigrama['pos'][word.lower()] += 1

        for word in posWordsBigramaTeste:
            frequenciaBigrama[word.lower()] += 1
            condicionalBigrama['pos'][word.lower()] += 1
         
        for word in negWordsUnigramaTeste:
            frequenciaUnigrama[word.lower()] += 1
            condicionalUnigrama['neg'][word.lower()] += 1

        for word in negWordsBigramaTeste:
            frequenciaBigrama[word.lower()] += 1
            condicionalBigrama['neg'][word.lower()] += 1
        featuresetUnigramaTeste,featuresetBigramaTeste = information_gain(frequenciaUnigrama,frequenciaBigrama,condicionalUnigrama,condicionalBigrama)
    # print len(featuresetUnigramaTeste),len(featuresetBigramaTeste)
    print featuresetUnigramaTeste
    return featuresetUnigramaTeste, featuresetBigramaTeste


def information_gain(frequenciaUnigrama,frequenciaBigrama,condicionalUnigrama,condicionalBigrama):
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
     
    bestUnigrama = sorted(word_scores_unigrama.iteritems(), key=lambda (w,s): s, reverse=True)[:4000]
    bestBigrama = sorted(word_scores_bigrama.iteritems(), key=lambda (w,s): s, reverse=True)[:4000]
    # print len(bestUnigrama),len(bestBigrama)
    # print bestUnigrama
    dicionarioUnigrama = [w for w, s in bestUnigrama]
    dicionarioBigrama = [w for w, s in bestBigrama]

    melhoresUnigramasPositivos = [(best_word_feats(frase, dicionarioUnigrama), 'no', indice) for frase,indice in zip(dicUnigramasPositivos,indicesPositivos) if (len(extraiFeatures(frase,dicionarioUnigrama)) > 0)]
    melhoresUnigramasNegativos = [(best_word_feats(frase, dicionarioUnigrama), 'yes', indice) for frase,indice in zip(dicUnigramasNegativos,indicesNegativos) if (len(extraiFeatures(frase,dicionarioUnigrama)) > 0)]
    melhoresBigramasPositivos = [(best_word_feats(frase, dicionarioBigrama), 'no', indice) for frase,indice in zip(dicBigramasPositivos,indicesPositivos) if (len(extraiFeatures(frase,dicionarioBigrama)) > 0)]
    melhoresBigramasNegativos = [(best_word_feats(frase, dicionarioBigrama), 'yes', indice) for frase,indice in zip(dicBigramasNegativos,indicesNegativos) if (len(extraiFeatures(frase,dicionarioBigrama)) > 0)]
    # print melhoresBigramasPositivos
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

regex_url = [
    r'<[^>]+>',  # HTML tags
    r'(?:@[\w_]+)',  # @-mentions
    r"(?:\#+[\w_]+[\w\'_\-]*[\w_]+)",  # hash-tags
    r'http[s]?://(?:[a-z]|[0-9]|[$-_@.&amp;+]|[!*\(\),]|(?:%[0-9a-f][0-9a-f]))+',  # URLs
]

remove_re = re.compile(r'(' + '|'.join(regex_url) + ')', re.VERBOSE | re.IGNORECASE)
tokens_re = re.compile(r'(' + '|'.join(regex_str) + ')', re.VERBOSE | re.IGNORECASE)
emoticon_re = re.compile(r'^' + emoticons_str + '$', re.VERBOSE | re.IGNORECASE)
# tknzr = TweetTokenizer()

datasets = ['OffComBR2']
radical = False
balanceado = False
ig = False
if sys.argv.count('d3'):
    datasets = ['OffComBR3']
if sys.argv.count('todos'):
    datasets = ['OffComBR2','OffComBR3','pt-de-BR2','pt-de-en-in-BR3']
if sys.argv.count('radical'):
    radical = True
if sys.argv.count('balanceado'):
    balanceado = True
if sys.argv.count('ig'):
    ig = True


stemmer=nltk.stem.RSLPStemmer()
# dataset = arff.load('OffComBR3.arff')
# incrementa_dataset(dataset)
# escreve_negativos(dataset)
# procura_indiceBR2_BR3()

for base_de_dados in datasets:
    if(base_de_dados == 'pt-de-BR2'):
        with open('negativosAlemaoPortuguesOFF2.txt','r') as f:
            data = f.readlines()
            frasesNegativas = [frases[:-1] for frases in data]
        radicais = [False]
        balanceados = [False]
        igs = [False]
    elif(base_de_dados == 'pt-de-en-in-BR3'):
        with open('negativosAleEspIngOFF3.txt','r') as f:
            data = f.readlines()
            frasesNegativas = [frases[:-1] for frases in data]
        radicais = [False]
        balanceados = [False]
        igs = [False]
    else:
        radicais = [False,True]
        balanceados = [False,True]
        igs = [False,True]
    nomeDoResultadoUnigrama = base_de_dados+'_1G'
    nomeDoResultadoBigrama = base_de_dados+'_2G'
    for radical in radicais:
        for balanceado in balanceados:
            for ig in igs:
                if(((radical and ig and balanceado) or (radical and ig) or (radical and balanceado) or (balanceado and ig))):
                    continue
                temIg = '+IG' if ig == True else ''
                temBalanceado = '+BA' if balanceado == True else ''
                temRadical = '+RA' if radical == True else ''
                if(ig == True and balanceado != True and radical != True):
                    nomeDoResultadoUnigrama = base_de_dados+'_1G'+temIg
                    nomeDoResultadoBigrama = base_de_dados+'_2G'+temIg
                if(balanceado == True and ig != True and radical != True):
                    nomeDoResultadoUnigrama = base_de_dados+'_1G'+temBalanceado
                    nomeDoResultadoBigrama = base_de_dados+'_2G'+temBalanceado
                if(radical == True and ig != True and balanceado != True):
                    nomeDoResultadoUnigrama = base_de_dados+'_1G'+temRadical
                    nomeDoResultadoBigrama = base_de_dados+'_2G'+temRadical
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
                palavrasPositivas = 0
                palavrasNegativas = 0
                indicesNegativos = []
                indicesPositivos = []
                for indice,x in enumerate(dataset):
                    unigramas = [term for term in preprocess(x[1].lower().encode('utf-8'),radical) if term not in stop]
                    bigrama = list(nltk.bigrams(unigramas))
                    bigramas = [' '.join(item) for item in bigrama]
                    if(len(unigramas) > 0):
                        if(x[0] == 'yes'):
                            indicesNegativos.append(indice)
                            palavrasNegativas += 1
                            negWordsUnigrama.extend(unigramas)
                            negWordsBigrama.extend(bigramas)
                            dicUnigramasNegativos.append(unigramas)
                            dicBigramasNegativos.append(bigramas)
                        else:
                            indicesPositivos.append(indice)
                            palavrasPositivas += 1
                            posWordsUnigrama.extend(unigramas)
                            posWordsBigrama.extend(bigramas)
                            dicUnigramasPositivos.append(unigramas)
                            dicBigramasPositivos.append(bigramas)

                if ig:
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
                    featuresetUnigrama,featuresetBigrama = information_gain(frequenciaUnigrama,frequenciaBigrama,condicionalUnigrama,condicionalBigrama)
                    # print featuresetUnigrama
                else:
                    todosBigramas = posWordsBigrama + negWordsBigrama
                    todosUnigramas = posWordsUnigrama + negWordsUnigrama
                    if(balanceado and (base_de_dados != 'pt-de-BR2') and (base_de_dados != 'pt-de-en-in-BR3')):
                        todosBigramas = posWordsBigrama[:len(negWordsBigrama)] + negWordsBigrama
                        todosUnigramas = posWordsUnigrama[:len(negWordsUnigrama)] + negWordsUnigrama
                    # posUnigrama = []
                    # negUnigrama = []
                    # posBigrama = []
                    # negBigrama = []
                    posUnigrama = [(extraiFeatures(frase,todosUnigramas),'no',indice) for frase,indice in zip(dicUnigramasPositivos,indicesPositivos)]
                    negUnigrama = [(extraiFeatures(frase,todosUnigramas),'yes',indice) for frase,indice in zip(dicUnigramasNegativos,indicesNegativos)]
                    posBigrama = [(extraiFeatures(frase,todosBigramas),'no',indice) for frase,indice in zip(dicBigramasPositivos,indicesPositivos)]
                    negBigrama = [(extraiFeatures(frase,todosBigramas),'yes',indice) for frase,indice in zip(dicBigramasNegativos,indicesNegativos)]
                    qtdPosUni = len(posUnigrama)
                    qtdNegUni = len(negUnigrama)
                    qtdPosBig = len(posBigrama)
                    qtdNegBig = len(negBigrama)
                    classes = ['no'] * qtdPosUni + ['yes'] * qtdNegUni
                    # print classes
                    featuresetUnigrama =  posUnigrama + negUnigrama
                    featuresetBigrama =  posBigrama + negBigrama
                  

                    # if (base_de_dados == 'pt-de-BR2'):
                    #     featuresetUnigrama = [(extraiFeatures(frase,todosUnigramas),'no',indice) for frase,indice in zip(dicUnigramasPositivos,indicesPositivos)] + [(extraiFeatures(frase,todosUnigramas),'yes',indice) for frase,indice in zip(dicUnigramasNegativos,indicesNegativos[:-419])]
                    #     featuresetBigrama = [(extraiFeatures(frase,todosBigramas),'no',indice) for frase,indice in zip(dicBigramasPositivos,indicesPositivos)] + [(extraiFeatures(frase,todosBigramas),'yes',indice) for frase,indice in zip(dicBigramasNegativos,indicesNegativos[:-419])]
                    #     featuresetUnigrama = featuresetUnigrama[:1247]
                    #     featuresetBigrama = featuresetBigrama[:1247]
                    # elif (base_de_dados == 'pt-de-en-in-BR3'):
                    #     featuresetUnigrama = [(extraiFeatures(frase,todosUnigramas),'no',indice) for frase,indice in zip(dicUnigramasPositivos,indicesPositivos)] + [(extraiFeatures(frase,todosUnigramas),'yes',indice) for frase,indice in zip(dicUnigramasNegativos,indicesNegativos[:-202])]
                    #     featuresetBigrama = [(extraiFeatures(frase,todosBigramas),'no',indice) for frase,indice in zip(dicBigramasPositivos,indicesPositivos)] + [(extraiFeatures(frase,todosBigramas),'yes',indice) for frase,indice in zip(dicBigramasNegativos,indicesNegativos[:-202])]
                    #     featuresetUnigrama = featuresetUnigrama[:1031]
                    #     featuresetBigrama = featuresetBigrama[:1031]

                # print len(featuresetUnigrama),len(featuresetBigrama)
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

                print(base_de_dados,radical,balanceado,ig)
                test_dataUni, test_dataBig = features_com_base_kaggle(radical,balanceado,ig) 
                for resultado in resultados:
                    indicesDoBalanceamento = []
                    featureset = []
                    seed = 42
                    random.Random(seed).shuffle(resultado['featureset'])
                    for res in resultado['featureset']:
                        indicesDoBalanceamento.append(res[2])
                        featureset.append(res[:2])
                    tamanho = len(featureset)
                    kf = KFold(n_splits=10)
                    soma = 0
                    precisionPositivoMedia = 0;
                    precisionNegativoMedia = 0;
                    recallPositivoMedia = 0;
                    recallNegativoMedia = 0;
                    fmeasurePositivoMedia = 0;
                    fmeasureNegativoMedia = 0;
                    ind = data_augmentation(base_de_dados)
                    # if(base_de_dados == 'pt-de-en-in-BR3' or base_de_dados == 'pt-de-BR2'):
                    #     featureset = aumenta_dados(indicesDoBalanceamento,ind,resultado['nome'])
                    # print(len(featureset))
                    classifier = nltk.NaiveBayesClassifier.train(featureset)
                    if (resultado['nome'] == 'Bigrama'):
                        test_data = test_dataBig
                    else:
                        test_data = test_dataUni
                    refsets = collections.defaultdict(set)
                    testsets = collections.defaultdict(set)
                    train_data = np.array(featureset)
                    test_data = np.array(test_data)
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
                    print((fmeasurePositivoMedia + fmeasureNegativoMedia)/2)
                    # for train, test in kf.split(featureset):
                    #     refsets = collections.defaultdict(set)
                    #     testsets = collections.defaultdict(set)
                    #     train_data = np.array(featureset)[train]
                    #     test_data = np.array(featureset)[test]
                    #     train_index = np.array(indicesDoBalanceamento)[train]
                    #     if(base_de_dados == 'pt-de-BR2' or base_de_dados == 'pt-de-en-in-BR3'):
                    #         train_augmentation = aumenta_dados(train_index,ind,resultado['nome'])
                    #         train_data = np.concatenate((train_data,train_augmentation),axis=0)
                    #     # random.Random(seed).shuffle(train_data)
                    #     classifier = nltk.NaiveBayesClassifier.train(train_data)
                    #     soma += nltk.classify.accuracy(classifier, test_data)
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
                    # average = soma/10
                    # precisionPositivoMedia = precisionPositivoMedia/10
                    # precisionNegativoMedia = precisionNegativoMedia/10
                    # recallPositivoMedia = recallPositivoMedia/10
                    # recallNegativoMedia = recallNegativoMedia/10
                    # fmeasurePositivoMedia = 2 * (precisionPositivoMedia*recallPositivoMedia)/(precisionPositivoMedia+recallPositivoMedia)
                    # fmeasureNegativoMedia = 2 * (precisionNegativoMedia*recallNegativoMedia)/(precisionNegativoMedia+recallNegativoMedia)
                    # if(base_de_dados == 'pt-de-en-in-BR3'):
                    #     palavrasNegativas = palavrasNegativas - 606
                    # elif(base_de_dados == 'pt-de-BR2'):
                    #     palavrasNegativas = palavrasNegativas - 419
                    # elif(balanceado):
                    #     palavrasPositivas = palavrasNegativas
                    # fmeasurePonderado = ((fmeasurePositivoMedia*palavrasPositivas)+(fmeasureNegativoMedia*(palavrasNegativas)))/(palavrasNegativas+palavrasPositivas)

                    # # mostInf = classifier.show_most_informative_features(10)
                    # if(base_de_dados != 'OffComBR3' and base_de_dados != 'OffComBR2'):
                    #     resultadosParaPlotar.append(fmeasurePonderado)
                    # elif(not((radical and ig and balanceado) or (radical and ig) or (radical and balanceado) or (balanceado and ig))):
                    #     resultadosParaPlotar.append(fmeasurePonderado)
 
# fig, ax = plt.subplots()
# fig.subplots_adjust(bottom=0.45)
# print(nomesDosResultadosParaPlotar)
# barra = ax.bar(nomesDosResultadosParaPlotar,resultadosParaPlotar, width = 0.8, color="blue")
# ax.set_yticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
# ax.set_xticklabels(nomesDosResultadosParaPlotar,rotation='vertical')
# ax.set_ylabel('F-Measure')
# ax.set_xlabel("Configuracoes")
# autolabel(barra)
# plt.savefig(base_de_dados+'SemStop')
# plt.show()
