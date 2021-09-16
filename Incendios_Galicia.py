#!/usr/bin/env python
# coding: utf-8

import streamlit as st
from PIL import Image
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import bokeh
from bokeh.plotting import figure
#import pickle
from sklearn.metrics import accuracy_score, auc, confusion_matrix, f1_score, precision_score, recall_score, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import multilabel_confusion_matrix, plot_confusion_matrix, classification_report

st.set_page_config( page_title="Incendios en Galicia",
                   #page_icon="🧊",
                   layout="wide",
                   initial_sidebar_state="expanded")



st.title('Análisis y predicción de la causa de incendios forestales en Galicia')
         
st.write('El presente proyecto tiene como objetivo el análisis de los incendios producidos en Galicia durante el periodo 2001 - 2015, ' +
         'así como realizar predicciones de la CAUSA de incendios con las características (datos) que el usuarios desea consultar.')
         

  
  
st.write('')
st.write('Para realizar una predicción, introduce los datos/características en los campos de la izquierda. El resultado de la predicción se mostrará a continuación.')

  
  

# Preprocesar el dataset (renombrar columnas, etc.)
url = 'https://raw.githubusercontent.com/LenaMorianu/Los-incendios-en-Galicia/main/dataset_modelo.csv'
df = pd.read_csv(url, encoding='ISO-8859-1')

df.drop(['Unnamed: 0'], axis=1, inplace=True)


df.rename(columns={'superficie':'Superficie_quemada',
                   'lat':'Latitud',
                   'lng':'Longitud',
                   'time_ctrl':'Tiempo_control',
                   'personal':'Personal',
                   'medios':'Medios',
                   'TMEDIA':'Temperatura_media',
                   'RACHA':'Racha',
                   'SOL':'Sol_horas',
                   'AÃ±o':'Ano',
                   'PRES_RANGE':'Presion',
                   'target':'Causa'}, inplace=True)



#Variables de predicción

st.sidebar.subheader('Valores para predicción:')

var1 = st.sidebar.number_input('Superficie_quemada', min_value=0.00, max_value=10000.00, step=100.00)
var2 = st.sidebar.number_input('Tiempo_control', min_value=0.00, max_value=1000.00, step=50.00)
var3 = st.sidebar.number_input('Medios', min_value=0, max_value=50000, step=5)
var4 = st.sidebar.number_input('Presion', min_value=0.00, max_value=10000.00, step=10.00) 
var5 = st.sidebar.number_input('Sol_horas', min_value=0.00, max_value=1000.00, step=3.00)
var6 = st.sidebar.number_input('Personal', min_value=0, max_value=10000, step=10)
var7 = st.sidebar.number_input('Racha', min_value=0.00, max_value=1000.00, step=10.00)
var8 = st.sidebar.number_input('Longitud', min_value=-10.00, max_value=-6.00, step=0.05)
var9 = st.sidebar.number_input('Latitud', min_value=41.00, max_value=44.00, step=0.05)
var10 = st.sidebar.number_input('Ano', min_value=2001, max_value=2015, step=1)
var11 = st.sidebar.number_input('Temperatura_media', min_value=-30.00, max_value=50.00, step=5.00)

st.write('')

boton_prediccion = st.sidebar.button('REALIZAR PREDICCIÓN')





# Crear los dataset de TRAIN y TEST
X_train, X_test, y_train, y_test = train_test_split(df.drop(['Causa'], axis = 1),
                                                    df['Causa'],
                                                    train_size   = 0.8,
                                                    random_state = 1234,
                                                    shuffle      = True,
                                                    stratify = df['Causa'])



modelo = RandomForestClassifier(bootstrap = True, 
                                criterion= 'entropy', 
                                max_depth=None, 
                                random_state=13,
                                n_estimators=150,
                                class_weight='balanced').fit(X_train, y_train)


y_pred = modelo.predict(X_test)
LABELS=['Intencionado', 'Causa desconocida', 'Negligencia', 'Fuego reproducido', ' Rayo']


st.write("La capacidad predictiva del modelo - TEST SCORING: {0:.2f} %".format(100 * modelo.score(X_test, y_test)))

#st.table(plot_confusion_matrix(modelo, X_test, y_test, normalize='true'))

st.write('RESULTADO PREDICCIÓN:')
st.write('__________________________________________________')


# Realizar la predicción
if boton_prediccion:
  values =[var1,var2,var3,var4,var5,var6,var7,var8,var9,var10,var11]
  columnas = list(df.columns.drop(['Causa']))
  df_pred = pd.DataFrame(values, columnas)
  pred = [list(df_pred[0])]
  result = modelo.predict(pred)
  prob = modelo.predict_proba(pred)
    
  if result == 1: st.button('CAUSA incendio: INTENCIONADO')
  if result == 2: st.button('CAUSA incendio: CAUSA DESCONOCIDA')
  if result == 3: st.button('CAUSA incendio: NEGLIGENCIA INTENCIONADO')
  if result == 4: st.button('CAUSA incendio: FUEGO REPRODUCIDO')
  if result == 5: st.button('CAUSA incendio: RAYO')


st.write('__________________________________________________')
st.write('')
st.write('')
st.write('')


#df.head()



df.Causa.replace((1,2,3,4,5),("intencionado","causa desconocida","negligencia","fuego reproducido","rayo"),inplace=True)

st.write('')
st.write('')
st.write('')
st.write('Ejemplo de observaciones del dataset de análisis:')
st.table(df.head())  

df.Causa.replace(("intencionado","causa desconocida","negligencia","fuego reproducido","rayo"), (1,2,3,4,5), inplace=True)

#st.table(classification_report(y_test, y_pred))

y_proba = pd.DataFrame(modelo.predict_proba(X_test))
y_proba.columns = y_proba.columns.map({0:'Intencionado',
                                       1:'Causa_desconocida',
                                       2:'Negligencia',
                                       3:'Fuego_reproducido',
                                       4:'Rayo'}).astype(str)

st.write('')
st.write('')
st.write('')
#st.write('La probabilidad de cada observación de pertenecer a cada CAUSA de incendio:')
#st.write('')

#y2 = y_proba.head(10)

#st.table(y2)
  
  
st.write('MATRIZ DE CONFUSIÓN:')

# Mostrar la matriz de confusión

conf_matrix = confusion_matrix(y_test, y_pred, normalize='true')  # confusion_matrix
fig = plt.figure(figsize=(10, 7))
sns.heatmap(conf_matrix, xticklabels=LABELS, yticklabels=LABELS, annot=True, fmt="f")
plt.title("\n Matriz de confusión \n")
plt.ylabel('REALIDAD \n')
plt.xlabel('\n PREDICCIÓN')
st.pyplot(fig)  
  
  
st.sidebar.markdown('__________________________________________________________________________')  

#image = Image.open('./images/IMG2.png')
#st.image(image, caption='UCM')          
    
st.sidebar.write('')
st.sidebar.write('Universidad Complutense de Madrid')
st.sidebar.write('MÁSTER BIG DATA & DATA SCIENCE')
st.sidebar.write('Madrid - Septiembre 2021')
st.sidebar.write('______________________________________________________________________')
st.sidebar.write('**AUTORES:**')
st.sidebar.markdown(' - Alejandra García Mosquera')
st.sidebar.markdown(' - Jorge Gómez Marco')
st.sidebar.markdown(' - Ana Hernández Villate')
st.sidebar.markdown(' - Alex Ilundain')
st.sidebar.markdown(' - Alicia María López Machado')
st.sidebar.markdown(' - Lena Morianu')
 
  
  
  
# Preguntar por el tamaño del dataset de TEST
#test_size = st.sidebar.slider(label = 'Elige el tamaño del dataset de TEST (%):',
#                              min_value=0,
#                              max_value=100,
#                              value=15,
#                              step=1)


