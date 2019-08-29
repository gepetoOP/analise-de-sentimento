import nltk
from nltk.metrics import BigramAssocMeasures
from sklearn.utils import shuffle

def balancear_dataset(dataset):
    positivos = dataset[dataset.Classificacao == 'no']
    negativos = dataset[dataset.Classificacao == 'yes']
    qtdNegativos = len(negativos)
    qtdPositivos = len(positivos)
    if qtdNegativos < qtdPositivos:
        tam = qtdNegativos
    else:
        tam = qtdPositivos
    positivos = shuffle(positivos)
    negativos = shuffle(negativos)
    return positivos[:tam].append(negativos[:tam])

def extrai_features(dataset,balancear=False):
    featureset = {}
    featureset['Unigrama'] = []
    featureset['Bigrama'] = []
    data = dataset.data
    if(balancear):
        data = balancear_dataset(data)
    classes = ['yes','no']
    estruturas = {'Unigrama': dataset.dicUnigrama,'Bigrama': dataset.dicBigrama}
    nova_data = data
    for classe in classes:
        for key,value in estruturas.items():
            conjunto = data[data.Classificacao == classe][key]
            for indice, frase in conjunto.iteritems():
                features = dict([(word, True) for word in value if word in frase])
                if(len(features) > 0):
                    featureset[key].append((features,classe,indice))
                else:
                    try:
                        nova_data = nova_data.drop(indice)
                    except:
                        continue
    dataset.setFeatureset(featureset)
    dataset.setData(nova_data)
    return dataset

def seleciona_features(dataset):
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

    for word, freq in frequenciaUnigrama.items():
        pos_score = BigramAssocMeasures.chi_sq(condicionalUnigrama['pos'][word],
            (freq, pos_word_count_unigrama), total_word_count_unigrama)
        neg_score = BigramAssocMeasures.chi_sq(condicionalUnigrama['neg'][word],
            (freq, neg_word_count_unigrama), total_word_count_unigrama)
        word_scores_unigrama[word] = pos_score + neg_score

    for word, freq in frequenciaBigrama.items():
        pos_score = BigramAssocMeasures.chi_sq(condicionalBigrama['pos'][word],
            (freq, pos_word_count_bigrama), total_word_count_bigrama)
        neg_score = BigramAssocMeasures.chi_sq(condicionalBigrama['neg'][word],
            (freq, neg_word_count_bigrama), total_word_count_bigrama)
        word_scores_bigrama[word] = pos_score + neg_score
    if(dataset.name == 'OffComBR3'):
        tamUni = 122
        tamBig = 103
    elif(dataset.name == 'OffComBR2'):
        tamUni = 250
        tamBig = 426
    bestUnigrama = sorted(word_scores_unigrama.items(), key=lambda w: w[1], reverse=True)[:tamUni]
    bestBigrama = sorted(word_scores_bigrama.items(), key=lambda w: w[1], reverse=True)[:tamBig]
    dicionarioUnigrama = [w for w, s in bestUnigrama]
    dicionarioBigrama = [w for w, s in bestBigrama]

    dataset.dicUnigrama = dicionarioUnigrama
    dataset.dicBigrama = dicionarioBigrama
    dataset = extrai_features(dataset)
    return dataset
