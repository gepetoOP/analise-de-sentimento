# -*- coding: utf-8 -*-

# This simple app uses the '/translate' resource to translate text from
# one language to another.

# This sample runs on Python 2.7.x and Python 3.x.
# You may need to install requests and uuid.
# Run: pip install requests uuid

import os, requests, uuid, json
import arff, numpy as np
from math import floor
# Checks to see if the Translator Text subscription key is available
# as an environment variable. If you are setting your subscription key as a
# string, then comment these lines out.
# If you want to set your subscription key as a string, uncomment the next line.
#subscriptionKey = 'put_your_key_here'

# If you encounter any issues with the base_url or path, make sure
# that you are using the latest endpoint: https://docs.microsoft.com/azure/cognitive-services/translator/reference/v3-0-translate
base_url = 'https://api.cognitive.microsofttranslator.com'
path = '/translate?api-version=3.0'
params = '&to=pt'
constructed_url = base_url + path + params

dataset = open('espanholOFF2.txt','r')
palavras = []
classes = []
for x in dataset:
	palavras.append({'text': x})
	# classes.append(x[0])
headers = {
    'Ocp-Apim-Subscription-Key': '9bd0a5266aac4af0b89a062e4af519b3',
    'Content-type': 'application/json',
    'X-ClientTraceId': str(uuid.uuid4())
}

# print(palavras)

# You can pass more than one object in body.
# body = [{
#     'text' : 'Hello World!',
# },
# {
#     'text' : 'Eai menor, tira a mão de mim',
# # }]
iteracoes = floor(len(palavras) / 25)
tamanho = len(palavras)
lista = {}
f = open('espanholParaPortuguesOFF2.txt','a')
# print(int(iteracoes))
# print(tamanho)
for i in range(int(iteracoes)):
	inicio = 25*(i)
	fim = 25*(i+1)
	request = requests.post(constructed_url, headers=headers, json=palavras[inicio:fim])
	response = request.json()
	print >> f, json.dumps(response, sort_keys=False, indent=4, separators=(',', ': '))
if(fim != tamanho):
	# print('argo')
	request = requests.post(constructed_url, headers=headers, json=palavras[fim:])
	response = request.json()
	print >> f, json.dumps(response, sort_keys=False, indent=4, separators=(',', ': '))
