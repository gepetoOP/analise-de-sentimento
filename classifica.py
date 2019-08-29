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
import operator
import collections
import pandas as pd
from io import StringIO
import arff, numpy as np
from unidecode import unidecode
import matplotlib.pyplot as plt
import concurrent.futures
from nltk.util import ngrams
from collections import Counter
#from googletrans import Translator
from nltk.tokenize import TweetTokenizer
from sklearn.model_selection import KFold
#from Aelius import Extras, Toqueniza, AnotaCorpus
from nltk.metrics.scores import precision, recall, f_measure
from unicodedata import normalize
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
import nltk.classify
from sklearn.svm import LinearSVC
from sklearn.utils import shuffle
from pre_processa import *
from extracao_de_caracteristicas import *

def aumenta_dados(indices,ind_balanceado,dataAugmentation,train_data):
    base_de_dados = dataAugmentation['nome_do_dataset']
    frasesNegativas = dataAugmentation['frasesNegativas']
    conjunto = dataAugmentation['tipo']
    dataset_original = dataAugmentation['dataset']
    feature_selection = dataAugmentation['feature_selection']
    radical = dataAugmentation['radical']
    frasesParaAumentar = []
    if(base_de_dados == 'OffComBR3'):
        for index in indices:
            if(ind_balanceado[index] != -1 and len(frasesNegativas[ind_balanceado[index]]) > 1):
                frasesParaAumentar.append(frasesNegativas[ind_balanceado[index]])
                frasesParaAumentar.append(frasesNegativas[ind_balanceado[index]+202])
                frasesParaAumentar.append(frasesNegativas[ind_balanceado[index]+404])
    else:
        for index in indices:
            if(ind_balanceado[index] != -1 and len(frasesNegativas[ind_balanceado[index]]) > 1):
                frasesParaAumentar.append(frasesNegativas[ind_balanceado[index]])
    data_antigo = dataset_original.data
    dataset = dataset_original
    unigramas = []
    bigramas = []
    dataset.data = pd.DataFrame()
    listaUnigrama = list(dataset.dicUnigrama)
    listaBigrama = list(dataset.dicBigrama)
    for i,frase in enumerate(frasesParaAumentar):
        unigramas = [term for term in preprocess(frase.lower().encode('utf-8'), radical) if term not in stop and len(term) > 1]
        bigrama = list(nltk.bigrams(unigramas))
        bigramas = [' '.join(item) for item in bigrama]
        bigramas.extend(unigramas)
        if len(bigramas) > 0:
            dataset.data = dataset.data.append(pd.Series({'Unigrama': unigramas, 'Bigrama': bigramas, 'Classificacao': 'yes'}, name = i))
            listaUnigrama.extend(unigramas)
            listaBigrama.extend(bigramas)
    dataset.dicUnigrama = set(listaUnigrama)
    dataset.dicBigrama = set(listaBigrama)
    if(feature_selection):
        dataset.data = pd.concat([data_antigo, dataset.data])
        dataset = seleciona_features(dataset)
    dataset = extrai_features(dataset)
    features = dataset.getFeatureset()
    # print(features)
    tipo = dataAugmentation['tipo']
    lista = []
    for feature in features[tipo]:
        lista.append(feature[:2])
    # featureset = lista + featureset
    # random.shuffle(featureset)
    return lista

def data_augmentation(dataset):
    with open('indicesNegativosOFF'+dataset[-1]+'.txt','r') as f:
        data = f.readlines()

        if(dataset == 'OffComBR2'):
            tamanho = 1250
        else:
            tamanho = 1033
        ind = [-1]*tamanho
        for indice,indiceNegativo in enumerate(data):
            ind[int(indiceNegativo[:-1])] = indice
    return ind

def instancia_dataset_arff(nome):
    raw_data = arff.load(nome + '.arff')
    linhas = [x for x in raw_data]
    df_data = pd.DataFrame(
        {
            'Text': [linha[1] for linha in linhas],
            'Classificacao': [linha[0] for linha in linhas]
        }
    )
    return df_data

def conta_positivos_e_negativos(featureset):
    qtdPositivos = 0
    qtdNegativos = 0
    tamanhos = {}
    tamanhos['Unigrama'] = {}
    tamanhos['Bigrama'] = {}
    tamanhos['Unigrama']['qtdNegativos'] = tamanhos['Bigrama']['qtdNegativos'] = tamanhos['Unigrama']['qtdPositivos'] = tamanhos['Bigrama']['qtdPositivos'] = 0
    tipos = ['Unigrama','Bigrama']
    for tipo in tipos:
        for features in featureset[tipo]:
            if(features[1] == 'yes'):
                tamanhos[tipo]['qtdNegativos'] += 1
            else:
                tamanhos[tipo]['qtdPositivos'] +=1
    return tamanhos

def cross_validation(featureset,k,tamanhos,dataAugmentation):
    soma = 0
    fmeasurePonderadoMedia = 0;
    nome_do_dataset = dataAugmentation['nome_do_dataset']
    oversampling = dataAugmentation['oversampling']
    ind = data_augmentation(nome_do_dataset)
    qtdNegativos = 0
    kf = KFold(n_splits=k)
    tam = 10
    fmeasureDesvio = [] 
    fmeasurePonderadoMedia = []
    random.Random().shuffle(featureset)
    if(oversampling):
        featuresetComIndice = featureset
        indicesDoBalanceamento = []
        featureset = []
        for features in featuresetComIndice:
            indicesDoBalanceamento.append(features[2])
            featureset.append(features[:2])
    for train, test in kf.split(featureset):
        qtdNegativos = tamanhos['qtdNegativos']
        refsets = collections.defaultdict(set)
        testsets = collections.defaultdict(set)
        train_data = np.array(featureset)[train]
        test_data = np.array(featureset)[test]
        if(oversampling):
            train_augmentation = aumenta_dados(indicesDoBalanceamento,ind,dataAugmentation,train_data)
            qtdNegativos += len(train_augmentation)
            train_data = np.concatenate((train_data,train_augmentation),axis=0)
            random.Random().shuffle(train_data)
        classifier = nltk.classify.SklearnClassifier(LinearSVC())
        # classifier = nltk.NaiveBayesClassifier.train(train_data)
        classifier.train(train_data)

        soma += nltk.classify.accuracy(classifier, test_data)
        for i, (feats, label) in enumerate(test_data):
            refsets[label].add(i)
            observed = classifier.classify(feats)
            testsets[observed].add(i)
        fmeasurePositivo = f_measure(refsets['no'], testsets['no'])
        fmeasureNegativo = f_measure(refsets['yes'], testsets['yes'])
        if(not fmeasurePositivo or not fmeasureNegativo):
            continue
        fmeasurePonderadoMedia.append(((fmeasurePositivo*tamanhos['qtdPositivos'])+(fmeasureNegativo*qtdNegativos))/(qtdNegativos+tamanhos['qtdPositivos']))
    # print(cont)
    fmeasurePonderado = np.mean(fmeasurePonderadoMedia)
    print(fmeasurePonderado)

def classifica(nome_dataset,frasesNegativas,configuracao):
    radical = configuracao['radical']
    ig = configuracao['ig']
    balanceado = configuracao['balanceado']

    dataAugmentation = {}
    dataAugmentation['frasesNegativas'] = frasesNegativas
    dataAugmentation['nome_do_dataset'] = nome_dataset
    dataAugmentation['oversampling'] = configuracao['oversampling']
    dataAugmentation['feature_selection'] = configuracao['ig']
    dataAugmentation['radical'] = radical

    data_atual = instancia_dataset_arff(nome_dataset)
    dataset = Dataset(data_atual,nome_dataset)
    dataset.gera_unigramas_e_bigramas(radical)

    if (ig):
        if(balanceado):
            dataset.data = balancear_dataset(dataset.data)
        dataset = seleciona_features(dataset)
    else:
        dataset = extrai_features(dataset, balanceado)
    featureset = dataset.getFeatureset()
    tamanhos = conta_positivos_e_negativos(featureset)
    dataAugmentation['dataset'] = dataset
    print(dataset.name)
    print(configuracao)
    for ngrama in featureset:
        dataAugmentation['tipo'] = ngrama
        if(not dataAugmentation['oversampling']):
            features = [f[:2] for f in featureset[ngrama]]
        else:
            features = featureset[ngrama]
        print(ngrama)
        cross_validation(features, 10, tamanhos[ngrama],dataAugmentation)
        # mostInf = classifier.show_most_informative_features(10)

class Dataset:
    def __init__(self,dataset,name):
        self.dataset = dataset
        self.name = name
        self.data = pd.DataFrame()
        self.dicUnigrama = []
        self.dicBigrama = []
        self.featureset = []
        self.positivo = dataset[dataset.Classificacao == 'no']['Text']
        self.indicesNegativos = []
        self.negativo = dataset[dataset.Classificacao == 'yes']['Text']

    def __add__(self, dataset2):
        nome = self.name + '+' + dataset2.name
        # dataset = dataset + dataset2
        positivo = list(self.positivo.values) + list(dataset2.positivo.values)
        classificacao_positiva = ['no'] * len(positivo)
        negativo = list(self.negativo.values) + list(dataset2.negativo.values)
        classificacao_negativa = ['yes'] * len(negativo)
        dataset = pd.DataFrame({'Text': (positivo + negativo), 'Classificacao': (classificacao_positiva + classificacao_negativa)})
        return Dataset(dataset,nome)

    def getFrasesPositivas(self):
        return self.positivo
    
    def getFrasesNegativas(self):
        return self.negativo

    def setFeatureset(self,featureset):
        self.featureset = featureset
    
    def getFeatureset(self):
        return self.featureset

    def setData(self,data):
        self.data = data

    def gera_unigramas_e_bigramas(self, radical):
        indices = []
        for i,frase in self.negativo.iteritems():
            unigramas = [term for term in preprocess(frase.lower().encode('utf-8'),radical) if term not in stop and len(term) > 1]
            bigrama = list(nltk.bigrams(unigramas))
            bigramas = [' '.join(item) for item in bigrama]
            bigramas.extend(unigramas)
            if len(unigramas) > 0:
                indices.append(i)
                self.data = self.data.append(pd.Series({'Unigrama': unigramas, 'Bigrama': bigramas, 'Classificacao': 'yes'}, name = i))
                self.dicUnigrama.extend(unigramas)
                self.dicBigrama.extend(bigramas)
                self.indicesNegativos.append(i)

        for i,frase in self.positivo.iteritems():
            unigramas = [term for term in preprocess(frase.lower().encode('utf-8'),radical) if term not in stop and len(term) > 1]
            bigrama = list(nltk.bigrams(unigramas))
            bigramas = [' '.join(item) for item in bigrama]
            bigramas.extend(unigramas)
            if len(unigramas) > 0:
                indices.append(i)
                self.data = self.data.append(pd.Series({'Unigrama': unigramas,'Bigrama': bigramas, 'Classificacao': 'no'}, name = i))
                self.dicUnigrama.extend(unigramas)
                self.dicBigrama.extend(bigramas)
        # if(self.name == 'kaggle'):
        self.dicUnigrama = set(self.dicUnigrama)
        self.dicBigrama = set(self.dicBigrama)

def main():
    datasets = ['OffComBR2','OffComBR3']
    configuracoes = [
    {
        'radical': False,
        'ig': False,
        'balanceado': False,
        'oversampling': False
    },
    {
        'radical': True,
        'ig': False,
        'balanceado': False,
        'oversampling': False
    },
    {
        'radical': False,
        'ig': True,
        'balanceado': False,
        'oversampling': False
    },
    {
        'radical': False,
        'ig': False,
        'balanceado': True,
        'oversampling': False
    },
    {
       'radical': False,
       'ig': False,
       'balanceado': False,
       'oversampling': True
    },
        {
        'radical': True,
        'ig': True,
        'balanceado': False,
        'oversampling': False
    },
    {
        'radical': True,
        'ig': False,
        'balanceado': True,
        'oversampling': False
    },
    {
       'radical': True,
       'ig': False,
       'balanceado': False,
       'oversampling': True
    },
     {
       'radical': False,
       'ig': True,
       'balanceado': True,
       'oversampling': False
    },
        {
        'radical': False,
        'ig': True,
        'balanceado': False,
        'oversampling': True
    },
     {
       'radical': True,
       'ig': True,
       'balanceado': False,
       'oversampling': True
    },
        {
        'radical': True,
        'ig': True,
        'balanceado': True,
        'oversampling': False
    },
    ]
    for dataset in datasets:
        for conf in configuracoes:
            frasesNegativas = []
            if(conf['oversampling'] == True):
                if(dataset == 'OffComBR2'):
                    with open('negativosAlemaoPortuguesOFF2.txt','r') as f:
                        data = f.readlines()
                        frasesNegativas = [frases[:-1] for frases in data]
                else:
                    with open('negativosAleEspIngOFF3.txt','r') as f:
                        data = f.readlines()
                        frasesNegativas = [frases[:-1] for frases in data]
            classifica(dataset, frasesNegativas,conf)

if __name__ == '__main__':
    main()
