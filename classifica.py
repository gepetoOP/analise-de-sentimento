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
# from scipy.io import loadarff 
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
import nltk.classify
from sklearn.svm import LinearSVC

punctuation = list(string.punctuation)
stop = stopwords.words('portuguese') + punctuation + ['rt', 'RT', 'via']

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

def aumenta_dados(train_index,ind_balanceado,dataAugmentation):
    base_de_dados = dataAugmentation['nome_do_dataset']
    frasesNegativas = dataAugmentation['frasesNegativas']
    conjunto = dataAugmentation['tipo']
    dataset = dataAugmentation['data']
    frasesParaAumentar = []
    if(base_de_dados == 'pt-de-en-in-BR3'):
        for index in train_index:
            if(ind_balanceado[index] != -1 and len(frasesNegativas[ind_balanceado[index]]) > 1):
                frasesParaAumentar.append(frasesNegativas[ind_balanceado[index]])
                # frasesParaAumentar.append(frasesNegativas[ind_balanceado[index]+202])
                # frasesParaAumentar.append(frasesNegativas[ind_balanceado[index]+404])
    else:
        for index in train_index:
            if(ind_balanceado[index] != -1 and len(frasesNegativas[ind_balanceado[index]]) > 1):
                frasesParaAumentar.append(frasesNegativas[ind_balanceado[index]])
    random.shuffle(frasesNegativas)
    unigramas = []
    bigramas = []
    dataset.data = pd.DataFrame()
    listaUnigrama = list(dataset.dicUnigrama)
    listaBigrama = list(dataset.dicBigrama)
    for i,frase in enumerate(frasesParaAumentar):
        unigramas = [term for term in preprocess(remover_acentos(frase).lower().encode('utf-8'),False) if term not in stop and len(term) > 1]
        bigrama = list(nltk.bigrams(unigramas))
        bigramas = [' '.join(item) for item in bigrama]
        dataset.data = dataset.data.append(pd.Series({'Unigrama': unigramas, 'Bigrama': bigramas, 'Classificacao': 'yes'}, name = i))
        listaUnigrama.extend(unigramas)
        listaBigrama.extend(bigramas)
    dataset.dicUnigrama = set(listaUnigrama)
    dataset.dicBigrama = set(listaBigrama)
    features = extraiFeatures(dataset)
    tipo = dataAugmentation['tipo']
    lista = []
    somaP = 0
    somaN = 0
    for feature in features[tipo]:
        lista.append(feature[:2])
        if(feature[2] == 'yes'):
            somaN += 1
        else:
            somaP += 1
    # print lista
    return lista

def data_augmentation(dataset):
    with open('indicesNegativosOFF'+dataset[-1]+'.txt','r') as f:
        data = f.readlines()
        ind = [-1]*1250
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

def remover_acentos(txt, codif='utf-8'):
    return normalize('NFKD', txt.decode(codif)).encode('ASCII', 'ignore')

def tokenize(s):
    return tokens_re.findall(s)

def best_word_feats(words, bestwords):
    return dict([(word, True) for word in words if word in bestwords])

def preprocess(s, radical,lowercase=True):
    frase = remove_re.sub('',s)
    tokens = tokenize(frase)
    if radical:
        return [stemmer.stem(token.lower()) for token in tokens]
    else:
        return [token.lower() for token in tokens]
    
    # return tokens

def balancear_dataset(dataset):
    positivos = dataset[dataset.Classificacao == 'no']
    negativos = dataset[dataset.Classificacao == 'yes']
    qtdNegativos = len(negativos)
    qtdPositivos = len(positivos)
    if qtdNegativos < qtdPositivos:
        tam = qtdNegativos
    else:
        tam = qtdPositivos
    return positivos[:tam].append(negativos[:tam])

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

def extraiFeatures(dataset,balancear=False):
    featureset = {}
    featureset['Unigrama'] = []
    featureset['Bigrama'] = []
    data = dataset.data
    if(balancear):
        data = balancear_dataset(data)
    classes = ['yes','no']
    estruturas = {'Unigrama': dataset.dicUnigrama,'Bigrama': dataset.dicBigrama}
    for classe in classes:
        for key,value in estruturas.iteritems():
            for indice,frase in enumerate(data[data.Classificacao == classe][key]):
                features = {}
                for word in value:
                    try:
                        if word in frase:
                            features["{}".format(word.encode('utf-8'))] = True
                    except Exception as inst:
                        print(inst)
                        continue
                if(len(features) > 0):
                    featureset[key].append((features,classe,indice))
    return featureset

def unique(lista):
    return list(dict.fromkeys(lista))

def comparar_datasets(treino,teste,configuracoes):
    ig = configuracoes['ig']
    radical = configuracoes['radical']
    balanceado = configuracoes['balanceado']
    if(treino == 'kaggle'):
        dataset_treino = pd.read_csv('Tweets_Mg.csv',encoding='utf-8')
        dataset_teste = instancia_dataset_arff(teste)
    else:
        dataset_treino = instancia_dataset_arff(treino)
        dataset_teste = pd.read_csv('Tweets_Mg.csv',encoding='utf-8')
    data_treino = Dataset(dataset_treino,treino)
    data_treino.gera_unigramas_e_bigramas(radical)
    data_teste = Dataset(dataset_teste,teste)
    data_teste.gera_unigramas_e_bigramas(radical)
    # print len(data_treino.positivo)
    if (ig):
        featureset_treino = information_gain(data_treino)
        featureset_teste = information_gain(data_teste)
    else:
        featureset_treino = extraiFeatures(data_treino, balanceado)
        featureset_teste = extraiFeatures(data_teste, balanceado)
    # tamanhos = conta_positivos_e_negativos(featureset)
    print('Treino:{}'.format(treino))
    print('Teste:{}'.format(teste))
    print configuracoes
    for ngrama in featureset_treino:
        features_treino = []
        features_teste = []
        # fmeasurePositivoMedia = 0
        # fmeasureNegativoMedia = 0
        # fmeasurePonderadoMedia = 0
        for it in featureset_treino[ngrama]:
            features_treino.append(it[:2])
        for it in featureset_teste[ngrama]:
            features_teste.append(it[:2])
        classifier = nltk.NaiveBayesClassifier.train(features_treino)
        refsets = collections.defaultdict(set)
        testsets = collections.defaultdict(set)
        train_data = np.array(features_treino)
        test_data = np.array(features_teste)
        acuracia = nltk.classify.accuracy(classifier, test_data)
        for i, (feats, label) in enumerate(test_data):
            refsets[label].add(i)
            observed = classifier.classify(feats)
            testsets[observed].add(i)
        fmeasurePositivo = f_measure(refsets['no'], testsets['no'])
        fmeasureNegativo = f_measure(refsets['yes'], testsets['yes'])
        print(ngrama)
        print(fmeasurePositivo,fmeasureNegativo,acuracia)
        # mostInf = classifier.show_most_informative_features(10)

def cross_validation(featureset,k,tamanhos,indicesDoBalanceamento,dataAugmentation):
    soma = 0
    precisionPositivoMedia = 0;
    precisionNegativoMedia = 0;
    recallPositivoMedia = 0;
    recallNegativoMedia = 0;
    fmeasurePositivoMedia = 0;
    fmeasureNegativoMedia = 0;
    fmeasurePonderadoMedia = 0;
    nome_do_dataset = dataAugmentation['nome_do_dataset']
    if(nome_do_dataset == 'pt-de-BR2'):
        positivo = 0
        for indice,features in enumerate(featureset):
            if(features[1] == 'yes'):
                continue
            else:
                positivo = indice
                break
        tamanhos['qtdNegativos'] = 419
        featureset = featureset[:419] + featureset[positivo:]
        ind = data_augmentation(nome_do_dataset)
        
    elif(nome_do_dataset == 'pt-de-en-in-BR3'):
        positivo = 0
        for indice,features in enumerate(featureset):
            if(features[1] == 'yes'):
                continue
            else:
                positivo = indice
                break
        tamanhos['qtdNegativos'] = 202
        featureset = featureset[:202] + featureset[positivo:]
        ind = data_augmentation(nome_do_dataset)
    qtdNegativos = 0
    kf = KFold(n_splits=k)
    tam = 10
    fmeasureDesvio = [] 
    # for i in range(tam):
    fmeasurePonderadoMedia = 0
    qtdNegativos = tamanhos['qtdNegativos']
    random.Random().shuffle(featureset)
    cont = 1
    for train, test in kf.split(featureset):
        qtdNegativos = tamanhos['qtdNegativos']
        refsets = collections.defaultdict(set)
        testsets = collections.defaultdict(set)
        train_data = np.array(featureset)[train]
        test_data = np.array(featureset)[test]
        train_index = np.array(indicesDoBalanceamento)[train]
        if(nome_do_dataset == 'pt-de-BR2' or nome_do_dataset == 'pt-de-en-in-BR3'):
            train_augmentation = aumenta_dados(train_index,ind,dataAugmentation)
            qtdNegativos += len(train_augmentation)
            train_data = np.concatenate((train_data,train_augmentation),axis=0)
            random.Random().shuffle(train_data)
        classifier = nltk.classify.SklearnClassifier(LinearSVC())
        # classifier = nltk.NaiveBayesClassifier.train(train_data)
        classifier.train(train_data)

        # soma += nltk.classify.accuracy(classifier, test_data)
        for i, (feats, label) in enumerate(test_data):
            refsets[label].add(i)
            observed = classifier.classify(feats)
            testsets[observed].add(i)
        # precisionPositivoMedia += precision(refsets['no'], testsets['no'])
        # precisionNegativoMedia += precision(refsets['yes'], testsets['yes'])
        # recallPositivoMedia += recall(refsets['no'], testsets['no'])
        # recallNegativoMedia += recall(refsets['yes'], testsets['yes'])
        # fmeasurePositivoMedia += f_measure(refsets['no'], testsets['no'])
        # fmeasureNegativoMedia += f_measure(refsets['yes'], testsets['yes'])
        fmeasurePositivo = f_measure(refsets['no'], testsets['no'])
        fmeasureNegativo = f_measure(refsets['yes'], testsets['yes'])
        if(not fmeasurePositivo or not fmeasureNegativo):
            continue
        fmeasurePonderadoMedia += ((fmeasurePositivo*tamanhos['qtdPositivos'])+(fmeasureNegativo*(qtdNegativos)))/(qtdNegativos+tamanhos['qtdPositivos'])
        cont += 1
    # average = soma/10
    # precisionPositivoMedia = precisionPositivoMedia/10
    # precisionNegativoMedia = precisionNegativoMedia/10
    # recallPositivoMedia = recallPositivoMedia/10
    # recallNegativoMedia = recallNegativoMedia/10
    # fmeasurePositivoMedia = 2 * (precisionPositivoMedia*recallPositivoMedia)/(precisionPositivoMedia+recallPositivoMedia)
    # fmeasureNegativoMedia = 2 * (precisionNegativoMedia*recallNegativoMedia)/(precisionNegativoMedia+recallNegativoMedia)
    fmeasurePonderado = fmeasurePonderadoMedia/cont
    print fmeasurePonderado
    # fmeasureDesvio.append(fmeasurePonderado)
    # dado = pd.DataFrame({'fmeasure': fmeasureDesvio})
    # print dado.fmeasure
    # print dado.fmeasure.mean()
    # print dado.fmeasure.std()

def classifica(dataset,frasesNegativas,configuracao):
    radical = configuracao['radical']
    ig = configuracao['ig']
    balanceado = configuracao['balanceado']
    dataAugmentation = {}
    dataAugmentation['frasesNegativas'] = frasesNegativas
    dataAugmentation['nome_do_dataset'] = dataset
    # dataset_kaggle = pd.read_csv('Tweets_Mg.csv',encoding='utf-8')
    # dataset_kaggle = Dataset(dataset_kaggle,'kaggle')
    if(dataset == 'kaggle'):
        data_atual = pd.read_csv('Tweets_Mg.csv',encoding='utf-8')
    else:
        data_atual = instancia_dataset_arff(dataset)
    dataset_primario = Dataset(data_atual,dataset)
    # dataset_atual = dataset_primario + dataset_kaggle
    dataset_atual = dataset_primario
    dataset_atual.gera_unigramas_e_bigramas(radical)
    if (ig):
        if(balanceado):
            data_atual = balancear_dataset(dataset_atual.data)
        featureset = information_gain(dataset_atual)
    else:
        featureset = extraiFeatures(dataset_atual, balanceado)
    tamanhos = conta_positivos_e_negativos(featureset)
    dataAugmentation['data'] = dataset_atual
    print dataset_atual.name
    print configuracao
    for ngrama in featureset:
        indicesDoBalanceamento = []
        dataAugmentation['tipo'] = ngrama
        features = []
        for it in featureset[ngrama]:
            features.append(it[:2])
            indicesDoBalanceamento.append(it[2])
        print ngrama
        cross_validation(features, 10, tamanhos[ngrama],indicesDoBalanceamento,dataAugmentation)
        # mostInf = classifier.show_most_informative_features(10)
 

def information_gain(dataset):
    frequenciaUnigrama = nltk.FreqDist()
    condicionalUnigrama = nltk.ConditionalFreqDist()
    dicionarioUnigrama = []

    frequenciaBigrama = nltk.FreqDist()
    condicionalBigrama = nltk.ConditionalFreqDist()
    dicionarioBigrama = []
    data = dataset.data
    for frase in data[data.Classificacao == 'no']['Unigrama']:
        for word in frase:
            frequenciaUnigrama[word.lower()] += 1
            condicionalUnigrama['pos'][word.lower()] += 1

    for frase in data[data.Classificacao == 'no']['Bigrama']:
        for word in frase:
            frequenciaBigrama[word.lower()] += 1
            condicionalBigrama['pos'][word.lower()] += 1
        
    for frase in data[data.Classificacao == 'yes']['Unigrama']:
        for word in frase:
            frequenciaUnigrama[word.lower()] += 1
            condicionalUnigrama['neg'][word.lower()] += 1

    for frase in data[data.Classificacao == 'yes']['Bigrama']:
        for word in frase:
            frequenciaBigrama[word.lower()] += 1
            condicionalBigrama['neg'][word.lower()] += 1
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
     
    bestUnigrama = sorted(word_scores_unigrama.iteritems(), key=lambda (w,s): s, reverse=True)[:2000]
    bestBigrama = sorted(word_scores_bigrama.iteritems(), key=lambda (w,s): s, reverse=True)[:4000]
    dicionarioUnigrama = [w for w, s in bestUnigrama]
    dicionarioBigrama = [w for w, s in bestBigrama]

    # melhoresUnigramasPositivos = [(best_word_feats(frase, dicionarioUnigrama), 'no') for frase in dataset.dicUnigrama if (len(best_word_feats(frase,dicionarioUnigrama)) > 0)]
    # melhoresUnigramasNegativos = [(best_word_feats(frase, dicionarioUnigrama), 'yes') for frase in dataset.dicUnigrama if (len(best_word_feats(frase,dicionarioUnigrama)) > 0)]
    # melhoresBigramasPositivos = [(best_word_feats(frase, dicionarioBigrama), 'no') for frase in dataset.dicBigrama if (len(best_word_feats(frase,dicionarioBigrama)) > 0)]
    # melhoresBigramasNegativos = [(best_word_feats(frase, dicionarioBigrama), 'yes') for frase in dataset.dicBigrama if (len(best_word_feats(frase,dicionarioBigrama)) > 0)]
    dataset.dicUnigrama = dicionarioUnigrama
    dataset.dicBigrama = dicionarioBigrama
    featureset = extraiFeatures(dataset)
    return featureset

def remover_duplicatas(dataset):
    frases = set()
    for frase in dataset[dataset.Classificacao == 'Positivo']['Text']:
        frases.add(remove_re.sub('',frase))
    qtdPositivos = len(frases)
    for frase in dataset[dataset.Classificacao == 'Negativo']['Text']:
        frases.add(remove_re.sub('',frase))
        # frases.add((remove_re.sub('',frase), dataset['Classificacao'][i]))
    qtdNegativos = len(frases) - qtdPositivos
    classificacao = (['Positivo'] * qtdPositivos) + (['Negativo'] * qtdNegativos) 
    return pd.DataFrame({'Text': list(frases), 'Classificacao': classificacao})
class Dataset:
    def __init__(self,dataset,name):
        self.dataset = dataset
        self.name = name
        self.data = pd.DataFrame()
        self.dicUnigrama = []
        self.dicBigrama = []
        if(name == 'kaggle'):
            dataset = remover_duplicatas(self.dataset)
            self.positivo = dataset[dataset.Classificacao == 'Positivo']['Text']
            self.negativo = dataset[dataset.Classificacao == 'Negativo']['Text']
        else:
            self.positivo = dataset[dataset.Classificacao == 'no']['Text']
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

    def gera_unigramas_e_bigramas(self, radical):
        indices = []
        for i,frase in self.negativo.iteritems():
            unigramas = [term for term in preprocess(remover_acentos(frase.lower().encode('utf-8')),radical) if term not in stop and len(term) > 1]
            bigrama = list(nltk.bigrams(unigramas))
            bigramas = [' '.join(item) for item in bigrama]
            if len(unigramas) > 0:
                indices.append(i)
                self.data = self.data.append(pd.Series({'Unigrama': unigramas, 'Bigrama': bigramas, 'Classificacao': 'yes'}, name = i))
                self.dicUnigrama.extend(unigramas)
                self.dicBigrama.extend(bigramas)

        for i,frase in self.positivo.iteritems():
            unigramas = [term for term in preprocess(remover_acentos(frase.lower().encode('utf-8')),radical) if term not in stop and len(term) > 1]
            bigrama = list(nltk.bigrams(unigramas))
            bigramas = [' '.join(item) for item in bigrama]
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
    },
    {
        'radical': True,
        'ig': False,
        'balanceado': False,
    },
    {
        'radical': False,
        'ig': True,
        'balanceado': False,
    },
    {
        'radical': False,
        'ig': False,
        'balanceado': True,
    },
    # {
    #     'radical': False,
    #     'ig': True,
    #     'balanceado': True,
    # },
    # {
    #     'radical': True,
    #     'ig': True,
    #     'balanceado': True,
    # }
    ]
    # datasets = ['pt-de-BR2','pt-de-en-in-BR3']
    # datasets = ['kaggle']
    for dataset in datasets:
        frasesNegativas = []
        if(dataset == 'pt-de-BR2'):
            with open('negativosAlemaoPortuguesOFF2.txt','r') as f:
                data = f.readlines()
                frasesNegativas = [frases[:-1] for frases in data]
        elif(dataset == 'pt-de-en-in-BR3'):
            with open('negativosAleEspIngOFF3.txt','r') as f:
                data = f.readlines()
                frasesNegativas = [frases[:-1] for frases in data]
        for conf in configuracoes:
            classifica(dataset, frasesNegativas,conf)
            # comparar_datasets(dataset,'kaggle',conf)

stemmer=nltk.stem.RSLPStemmer()
main()
