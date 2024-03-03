import sys, warnings

import nltk
nltk.download('punkt')

from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from nltk.tokenize import word_tokenize
from numpy import *
from urllib.parse import unquote

import numpy as np
import pandas as pd
import csv
import urllib.parse as parse
import pickle

from flask import Flask, request


# Metodo para convertir a un array de cadenas de consulta a un conjunto de características (features)
def getVec(text):
    tagged_data = [TaggedDocument(words=word_tokenize(_d.lower()), tags=[str(i)]) for i, _d in enumerate(text)]
    # max_epochs = 25
    max_epochs = 1
    vec_size = 20
    alpha = 0.025

    model = Doc2Vec(vector_size=vec_size,
                alpha=alpha, 
                min_alpha=0.00025,
                min_count=1,
                dm=1)
    model.build_vocab(tagged_data)
    print("Creando modelo de vector de muestra...")
    features = []
    for epoch in range(max_epochs):
        # print('Doc2Vec Iteracion # {0}'.format(epoch +1))
        print(f"Doc2Vec Iteracion # {epoch + 1}/{max_epochs}")
        # print("*", sep=' ', end='', flush=True)
        model.random.seed(42)
        model.train(tagged_data,
                total_examples=model.corpus_count,
                epochs=model.epochs)
        # ajuste la tasa de aprendizaje
        model.alpha -= 0.0002
        # Tasa de aprendisaje, sin decrees
        model.min_alpha = model.alpha
    model.save("lib/d2v.model")
    print("")
    print("Modelo Guardado")
    for i, line in enumerate(text):
        featureVec = [model.dv[i]]
        lineDecode = unquote(line)
        lineDecode = lineDecode.replace(" ", "")
        lowerStr = str(lineDecode).lower()
        #print("X"+str(i)+"=> "+line)

        # Podemos obtener mas features
        # https://websitesetup.org/javascript-cheat-sheet/
        # https://owasp.org/www-community/xss-filter-evasion-cheatsheet
        # https://html5sec.org/
        
        # Agregar características para contar etiquetas HTML maliciosas
        feature1 = int(lowerStr.count('<link'))
        feature1 += int(lowerStr.count('<object'))
        feature1 += int(lowerStr.count('<form'))
        feature1 += int(lowerStr.count('<embed'))
        feature1 += int(lowerStr.count('<ilayer'))
        feature1 += int(lowerStr.count('<layer'))
        feature1 += int(lowerStr.count('<style'))
        feature1 += int(lowerStr.count('<applet'))
        feature1 += int(lowerStr.count('<meta'))
        feature1 += int(lowerStr.count('<img'))
        feature1 += int(lowerStr.count('<iframe'))
        feature1 += int(lowerStr.count('<input'))
        feature1 += int(lowerStr.count('<body'))
        feature1 += int(lowerStr.count('<video'))
        feature1 += int(lowerStr.count('<button'))
        feature1 += int(lowerStr.count('<math'))
        feature1 += int(lowerStr.count('<picture'))
        feature1 += int(lowerStr.count('<map'))
        feature1 += int(lowerStr.count('<svg'))
        feature1 += int(lowerStr.count('<div'))
        feature1 += int(lowerStr.count('<a'))
        feature1 += int(lowerStr.count('<details'))
        feature1 += int(lowerStr.count('<frameset'))
        feature1 += int(lowerStr.count('<table'))
        feature1 += int(lowerStr.count('<comment'))
        feature1 += int(lowerStr.count('<base'))
        feature1 += int(lowerStr.count('<image'))

        # Agregamos características para contar métodos/eventos maliciosos de JAVSCRIPT
        feature2 = int(lowerStr.count('exec'))
        feature2 += int(lowerStr.count('fromcharcode'))
        feature2 += int(lowerStr.count('eval'))
        feature2 += int(lowerStr.count('alert'))
        feature2 += int(lowerStr.count('getelementsbytagname'))
        feature2 += int(lowerStr.count('write'))
        feature2 += int(lowerStr.count('unescape'))
        feature2 += int(lowerStr.count('escape'))
        feature2 += int(lowerStr.count('prompt'))
        feature2 += int(lowerStr.count('onload'))
        feature2 += int(lowerStr.count('onclick'))
        feature2 += int(lowerStr.count('onerror'))
        feature2 += int(lowerStr.count('onpage'))
        feature2 += int(lowerStr.count('confirm'))
        feature2 += int(lowerStr.count('marquee'))
        # Agregamos fetaure para ".js" contador
        feature3 = int(lowerStr.count('.js'))
        # Agregamos fetaure para "javascript" contador
        feature4 = int(lowerStr.count('javascript'))
        # Agregamos característica para la longitud del string
        feature5 = int(len(lowerStr))
        # Agregamos fetaure para "<script" contador
        feature6 = int(lowerStr.count('<script'))
        feature6 += int(lowerStr.count('&lt;script'))
        feature6 += int(lowerStr.count('%3cscript'))
        feature6 += int(lowerStr.count('%3c%73%63%72%69%70%74'))
        # Agregamos fetaures para caracteres especiales contador
        feature7 = int(lowerStr.count('&'))
        feature7 += int(lowerStr.count('<'))
        feature7 += int(lowerStr.count('>'))
        feature7 += int(lowerStr.count('"'))
        feature7 += int(lowerStr.count('\''))
        feature7 += int(lowerStr.count('/'))
        feature7 += int(lowerStr.count('%'))
        feature7 += int(lowerStr.count('*'))
        feature7 += int(lowerStr.count(';'))
        feature7 += int(lowerStr.count('+'))
        feature7 += int(lowerStr.count('='))
        feature7 += int(lowerStr.count('%3C'))
        # Agregamos fetaure para "http" contador
        feature8 = int(lowerStr.count('http'))
        
        # Agregamos a features
        featureVec = np.append(featureVec,feature1)
        #featureVec = np.append(featureVec,feature2)
        featureVec = np.append(featureVec,feature3)
        featureVec = np.append(featureVec,feature4)
        featureVec = np.append(featureVec,feature5)
        featureVec = np.append(featureVec,feature6)
        featureVec = np.append(featureVec,feature7)
        #featureVec = np.append(featureVec,feature8)
        #print(featureVec)
        features.append(featureVec)
    return features



testXSS = []
testNORM = []
X_temp = []
X = []
y = []
xssnum = 0
notxssnum = 0


# # Modelos guardados en local
filename1 = 'lib/DecisionTreeClassifier.sav'

filename2 = 'lib/SVC.sav'

filename3 = 'lib/GaussianNB.sav'

filename4 = 'lib/KNeighborsClassifier.sav'

filename5 = 'lib/RandomForestClassifier.sav'

filename6 = 'lib/MLPClassifier.sav'


# Leemos modelos desde Archivo Local
print("Leemos modelos...")
loaded_model1 = pickle.load(open(filename1, 'rb'))
loaded_model2 = pickle.load(open(filename2, 'rb'))
loaded_model3 = pickle.load(open(filename3, 'rb'))
loaded_model4 = pickle.load(open(filename4, 'rb'))
loaded_model5 = pickle.load(open(filename5, 'rb'))
loaded_model6 = pickle.load(open(filename6, 'rb'))


#print(Xnew)


def check_xss(testXSS):

    # Predicciones
    Xnew = getVec(testXSS)
    # Hacer predicciones

    #1 DecisionTreeClassifier
    ynew1 = loaded_model1.predict(Xnew)
    #2 SVC
    ynew2 = loaded_model2.predict(Xnew)
    #3 GaussianNB
    ynew3 = loaded_model3.predict(Xnew)
    #4 KNeighborsClassifier
    ynew4 = loaded_model4.predict(Xnew)
    #5 RandomForestClassifier
    ynew5 = loaded_model5.predict(Xnew)
    #6 MLPClassifier
    ynew6 = loaded_model6.predict(Xnew)


    # Resultados de los datos de entrada que fueron predictas
    xssCount = 0 
    notXssCount = 0
    for i in range(len(Xnew)):
        score = ((.175*ynew1[i])+(.15*ynew2[i])+(.05*ynew3[i])+(.075*ynew4[i])+(.25*ynew5[i])+(.3*ynew6[i]))
        # print(ynew1[i])
        # print(ynew2[i])
        # print(ynew3[i])
        # print(ynew4[i])
        # print(ynew5[i])
        # print(ynew6[i])
        # print(score)
        return_ml_xss = {}
        return_ml_xss['score'] = score
        if score >= .5:
            # print("\033[1;31;1mXSS\033[0;0m => "+testXSS[i])
            return_ml_xss['result'] = True
        else:
            return_ml_xss['result'] = False
            # print("\033[1;32;1mNO XSS\033[0;0m => "+testXSS[i])
        return return_ml_xss

# testXSS = [
#             # '<script>alert(\'xss\')</script><script><script>',
#             'richard',
#             ]

# if check_xss(testXSS)['result']:
#     print("XSS")
# else:
#     print("NO XSS")

app = Flask(__name__)

# Página con el formulario
@app.route('/')
def formulario():
    return '''
        <form action="/mostrar_datos" method="get">
            <label for="nombre">Nombre:</label><br>
            <input type="text" id="nombre" name="nombre"><br><br>
            <input type="submit" value="Enviar">
        </form> 
    '''

# Ruta que muestra los datos enviados desde el formulario
@app.route('/mostrar_datos', methods=['GET'])
def mostrar_datos():
    nombre = request.args.get('nombre')
    # testXSS = [nombre]
    result = check_xss([nombre])
    score = result['score']
    if result['result']:
        result_xss = f'<b><p style="color:red;">Resultado: XSS</p><p>Score: {score}</p></b>'
    else:
        result_xss = f'<b><p style="color:green;">Resultado: NO XSS </p><p>Score: {score}</p></b>'
    return f'<h1>Datos del Formulario </h1><p>Nombre: {nombre}</p>{result_xss}'

if __name__ == '__main__':
    app.run(debug=True)
