#!/usr/bin/env python
# coding: utf-8

# <h1>PRUEBA PRÁCTICA</h1>
# <h3>YOUNG PROFESSIONAL DATA</h3> <b>Andrea Carolina Sánchez Valdés</b>

# <h3>DATA SCIENCE METODOLOGY</h3>
# Para el desarrollo de la prueba se seguirá el ciclo de la metodologia de data science indicado en la imagen

# <img src = "http://www.ibmbigdatahub.com/sites/default/files/figure01_revised.jpg">

# <b>1. Business Understanding</b>
# 
# Se requiere identificar patrones o correlaciones entre variables

# <b>2. Analytic Approach</b>
# 
# El tipo de pregunta define el tipo de tratamiento para el problema, en este caso se requiere 
# descubrir relaciones entre variables, por lo tanto se puede usar un modelo descriptivo.

# <b>4. Data Requirements</b>
# 
# Para respoder o solucionar el problema se necesitan datos de los clientes que describan su comportamiento

# <b>4. Data Collection</b>
# 
# Identificar las fuentes de datos y como se obtuvieron

# In[ ]:


#Importar librerías
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import os
from datetime import date


# In[ ]:


#Leer datos BASE_ID.txt
id_path=("C://Users/hecansaga/Desktop/YPD/BASE_ID.txt")
df_id = pd.read_table(id_path)
df_id.head()


# In[ ]:


#Leer datos BASE_MOVIMIENTOS.txt
bm_path=("C://Users/hecansaga/Desktop/YPD/BASE_MOVIMIENTOS.txt")                   
df_bm = pd.read_fwf(bm_path)
df_bm.head()


# <b>5. Data Understanding</b>
# 
# Identificar las fuentes de datos y como se obtuvieron

# In[ ]:


#Tipos de variables en la base de movimientos
df_bm.dtypes


# In[ ]:


#Tipos de variables en la base id
df_id.dtypes


# <b>6. Data Preparation</b>
# 
# Transformar y limpiar los datos

# In[ ]:


#Correción y limpieza de datos
df_bm.replace(regex=True,inplace=True,to_replace=r'\$ ',value=r'') 
df_bm.replace(regex=True,inplace=True,to_replace=r'\.',value=r'')
df_bm.replace(regex=True,inplace=True,to_replace=r'\,',value=r'.')


# In[ ]:


#Configurar correctamente los tipos de variables de la base de movimientos
df_bm['SALDO_FONDOS']=df_bm['SALDO_FONDOS'].astype("float64")
df_bm['SALDO_CREDITO1']=df_bm['SALDO_CREDITO1'].astype("float64")
df_bm['SALDO_CREDITO2']=df_bm['SALDO_CREDITO2'].astype("float64")
df_bm['SALDO_ACTIVO']=df_bm['SALDO_ACTIVO'].astype("float64")
df_bm['SALDO_PASIVO']=df_bm['SALDO_PASIVO'].astype("float64")


# In[ ]:


#verificar los nuevos tipos de variables
df_bm.dtypes


# In[ ]:


#eliminar las datos de horas en la variable FECHA_INFORMACION
df_bm['FECHA_INFORMACION'].replace(regex=True,inplace=True,to_replace=r'\:00',value=r'')


# In[ ]:


#Reemplazar valores de meses en letras por números en FECHA INFORMACION
df_bm['FECHA_INFORMACION'].replace(regex=True,inplace=True,
                                   to_replace=['JAN','FEB','MAR','APR','MAY','JUN','JUL','AUG','SEP','OCT','NOV','DEC'],
                                   value=['/01/','/02/','/03/','/04/','/05/','/06/','/07/','/08/','/09/','/10/','/11/','/12/'])


# In[ ]:


#Convertir la variable FECHA INFORMACION  a tipo fecha
df_bm['FECHA_INFORMACION']=pd.to_datetime(df_bm['FECHA_INFORMACION'])


# In[ ]:


#Medidas estadisticas básicas para identificar los cambios en las variables
df_bm.describe()


# In[ ]:


#Observar la estructura de los datos corregidos de la base de movimientos
df_bm.head()


# In[ ]:


#Observar la estructura de los datos corregidos de la base id
df_id.head()


# In[ ]:


#Renombrar columna fuga por FUGA y CLIENTE CC por ID
df_id.rename(columns={'fuga':'FUGA','CLIENTE_CC':'ID'},inplace=True)


# In[ ]:


#Correción y limpieza de datos
df_id.replace(regex=True,inplace=True,to_replace=r'\$ ',value=r'') 
df_id.replace(regex=True,inplace=True,to_replace=r'\.',value=r'')
df_id.replace(regex=True,inplace=True,to_replace=r'\,',value=r'.')


# In[ ]:


#Reemplazar los datos NaN por 0s en las variables FUGA y MES DE FUGA
df_id['FUGA']=df_id['FUGA'].fillna(0)
df_id['MES_DE_FUGA']=df_id['MES_DE_FUGA'].fillna(0)


# In[ ]:


#Identificar valores unicos en la variable SEXO
df_id['SEXO'].unique()


# In[ ]:


#Unificar los valores en la variable SEXO
df_id['SEXO'].replace(inplace=True,to_replace=['F', 'HOMBRE', 'M', 'Hombre', 'mujer', 'femenino', 'masculino',
       'FEMENINO', 'Mujer', 'varón', 'Masc', 'MUJER'],
                      value=['FEMENINO', 'MASCULINO', 'MASCULINO', 'MASCULINO', 'FEMENINO', 'FEMENINO', 'MASCULINO',
       'FEMENINO', 'FEMENINO', 'MASCULINO', 'MASCULINO', 'FEMENINO'])


# In[ ]:


#Identificar valores unicos en la variable ESTADO CIVIL
df_id['ESTADO_CIVIL'].unique()


# In[ ]:


#Identificar valores unicos en la variable SITUACION LABORAL
df_id['SITUACION_LABORAL'].unique()


# In[ ]:


#Unificar los valores en la variable SITUACION LABORAL
df_id['SITUACION_LABORAL'].replace(inplace=True,to_replace=['otros', 'Contrato fijo', 'OTROS', 'contrato autonomo',
       ' desconocido   ', 'CONTRATO AUTONOMO', 'CONTRATO FIJO', 'CONTRATO TEMPORAL', 'temporal     ', 'SIN CLASIFICAR'],
                                   value=['OTROS', 'CONTRATO FIJO', 'OTROS', 'CONTRATO AUTONOMO',
       'DESCONOCIDO', 'CONTRATO AUTONOMO', 'CONTRATO FIJO', 'CONTRATO TEMPORAL', 'CONTRATO TEMPORAL', 'SIN CLASIFICAR'])


# In[ ]:


#Reemplazar valores de meses en letras por números en FECHA ALTA
df_id['FECHA_ALTA'].replace(regex=True,inplace=True,
                            to_replace=['ene','feb','mar','abr','may','jun','jul','ago','sep','oct','nov','dic'],
                            value=['01','02','03','04','05','06','07','08','09','10','11','12'])


# In[ ]:


#Extraer mes de FECHA ALTA
df_mesid1=df_id['FECHA_ALTA'].str[:2].astype(object)
df_mesid1.head()


# In[ ]:


#Extraer año de FECHA ALTA
df_añoid1=df_id['FECHA_ALTA'].str[4:].astype(object)
df_añoid1.head()


# In[ ]:


#Extraer día de FECHA ALTA
df_diaid1=df_id['FECHA_ALTA'].str[2:4:].astype(object)
df_diaid1.head()


# In[ ]:


#Extraer mes de FECHA NACIMIENTO
df_mesid2=df_id['FECHA_NACIMIENTO'].str[4:6:].astype(object)
df_mesid2.head()


# In[ ]:


#Extraer año de FECHA NACIMIENTO
df_añoid2=df_id['FECHA_NACIMIENTO'].str[:4].astype(object)
df_añoid2.head()


# In[ ]:


#Extraer día de FECHA NACIMIENTO
df_diaid2=df_id['FECHA_NACIMIENTO'].str[6:].astype(object)
df_diaid2.head()


# In[ ]:


#Construir nuevavamente FECHA ALTA
df_id['FECHA_ALTA'] = df_diaid1+'/'+df_mesid1+'/'+df_añoid1
df_id['FECHA_ALTA'].head()


# In[ ]:


#Construir nuevavamente FECHA NACIMIENTO
df_id['FECHA_NACIMIENTO'] = df_diaid2+'/'+df_mesid2+'/'+df_añoid2
df_id['FECHA_NACIMIENTO'].head()


# In[ ]:


#Identificar dato erroneo
df_len=df_id['FECHA_NACIMIENTO'].str.len()
df_len.head()


# In[ ]:


#Ubicación del dato erroreo
df_len.idxmax()


# In[ ]:


#Reemplazar valor del dato erroneo
df_id['FECHA_NACIMIENTO'].replace(inplace=True,to_replace='1-01/-0/0001',value='NaN')


# In[ ]:


#Configurar los tipos de variables de la base id
df_id['ID']=df_id['ID'].astype('float64')
df_id['ID']=df_id['ID'].astype('int64')
df_id['FECHA_ALTA']=pd.to_datetime(df_id['FECHA_ALTA'])
df_id['FECHA_NACIMIENTO']=pd.to_datetime(df_id['FECHA_NACIMIENTO'])
df_id['FUGA']=df_id['FUGA'].astype('bool')
df_id['MES_DE_FUGA']=df_id['MES_DE_FUGA'].astype('int64')
df_id.dtypes


# In[ ]:


#Organizar la base de movimientos por orden ascendente según la variable ID y luego por FECHA INFORMACION
df_bm=df_bm.sort_values(by=['ID','FECHA_INFORMACION'])
df_bm.head()


# In[ ]:


#Unir las bases id y movimientos en una que mantiene todos los datos de la base de movimientos
df_join=pd.merge(df_bm,df_id[['ID','FUGA','MES_DE_FUGA']],on='ID',how='left')
df_join.head()


# In[ ]:


#Verificación de la nueva base de datos
df_join.tail()


# In[ ]:


#Promedio abono nomina por trimestre

count_row=df_join.shape[0]

for i in range(0, count_row):
    a=df_join.at[i,'MONTO_ABONOS_NOMINA']
    b=df_join.at[i+1,'MONTO_ABONOS_NOMINA']
    c=df_join.at[i+2,'MONTO_ABONOS_NOMINA']
    prom=(a+b+c)/3
    df_join.at[i+2,'PROM_TRIM_ABONOS_NOMINA']=prom
    i=i+1
    if (i==31438):
        break


# In[ ]:


#Eliminar datos de promedio trimestre que no son correctos
ids=df_join.groupby('ID').size()

for i in range(2,2501):      
    countid=ids.at[i,]
    j=np.where(df_join["ID"] == i)[0]
    k=j[0]
    df_join.at[k,'PROM_TRIM_ABONOS_NOMINA']=0.0
    df_join.at[k+1,'PROM_TRIM_ABONOS_NOMINA']=0.0
    i=i+1
    if i>2501:
        break


# In[ ]:


#Promedio saldo ahorros

count_row=df_join.shape[0]

for i in range(0, count_row):
    a=df_join.at[i,'SALDO_AHORROS']
    b=df_join.at[i+1,'SALDO_AHORROS']
    c=df_join.at[i+2,'SALDO_AHORROS']
    prom=(a+b+c)/3
    df_join.at[i+2,'PROM_TRIM_SALDO_AHORROS']=prom
    i=i+1
    if (i==31438):
        break


# In[ ]:


#Eliminar datos de promedio trimestre que no son correctos
ids=df_join.groupby('ID').size()

for i in range(2,2501):      
    countid=ids.at[i,]
    j=np.where(df_join["ID"] == i)[0]
    k=j[0]
    df_join.at[k,'PROM_TRIM_SALDO_AHORROS']=0.0
    df_join.at[k+1,'PROM_TRIM_SALDO_AHORROS']=0.0
    i=i+1
    if i>2501:
        break


# In[ ]:


#Promedio saldo credito

count_row=df_join.shape[0]

for i in range(0, count_row):
    a=df_join.at[i,'SALDO_CREDITO1']
    b=df_join.at[i+1,'SALDO_CREDITO1']
    c=df_join.at[i+2,'SALDO_CREDITO1']
    d=df_join.at[i,'SALDO_CREDITO2']
    e=df_join.at[i+1,'SALDO_CREDITO2']
    f=df_join.at[i+2,'SALDO_CREDITO2']
    prom=(a+b+c+d+e+f)/6
    df_join.at[i+2,'PROM_TRIM_SALDO_CREDITO']=prom
    i=i+1
    if (i==31438):
        break


# In[ ]:


#Eliminar datos de promedio trimestre que no son correctos
ids=df_join.groupby('ID').size()

for i in range(2,2501):      
    countid=ids.at[i,]
    j=np.where(df_join["ID"] == i)[0]
    k=j[0]
    df_join.at[k,'PROM_TRIM_SALDO_CREDITO']=0.0
    df_join.at[k+1,'PROM_TRIM_SALDO_CREDITO']=0.0
    i=i+1
    if i>2501:
        break


# In[ ]:


#Verificación
df_join.head()


# In[ ]:


#Generar promedio de saldos y abono nomina para los trimestres corridos
for i in range(0,2500):
    j=np.where(df_join["ID"] == i+1)[0]
    y=tuple(j)
    df_i=df_join.loc[y,['ID','PROM_TRIM_ABONOS_NOMINA','PROM_TRIM_SALDO_AHORROS','PROM_TRIM_SALDO_CREDITO']]
    df_id.at[i,'PROM_TA_ABONO_NOMINA']=df_i["PROM_TRIM_ABONOS_NOMINA"].mean()
    df_id.at[i,'PROM_TA_SALDO_AHORROS']=df_i["PROM_TRIM_SALDO_AHORROS"].mean()
    df_id.at[i,'PROM_TA_SALDO_CREDITO']=df_i["PROM_TRIM_SALDO_CREDITO"].mean()
    i=i+1
    if i>2501:
        break


# In[ ]:


#Generar edad de los clientes
date='2017-12-01'
date=pd.to_datetime(date)
for i in range(0,2500):
    df_id.loc[i,'EDAD']= (date- df_id.at[i,'FECHA_NACIMIENTO']) / 365
    i=i+1
    if i>2501:
        break


# In[ ]:


df_id['EDAD']=df_id['EDAD'].dt.days


# In[ ]:


#Generar tiempo de alta de los clientes
date='2017-12-01'
date=pd.to_datetime(date)
for i in range(0,2500):
    df_id.loc[i,'TIEMPO_ALTA']= (date- df_id.at[i,'FECHA_ALTA']) / 365
    i=i+1
    if i>2501:
        break


# In[ ]:


df_id['TIEMPO_ALTA']=df_id['TIEMPO_ALTA'].dt.days


# <b>7. Modelling</b>
# 
# Estudiar los datos de manera predictiva y descriptiva

# In[ ]:


#Cantidad de datos de cada tipo
print(df_id.groupby('FUGA').size())


# In[ ]:


#Identificar correlaciones de la base id
df_id.corr()


# In[ ]:


#Identificar correlaciones de la base id mediante un heatmap
sns.heatmap(df_id.corr(), square=True, annot=True)


# In[ ]:


#Histograma de las variables
df_id.drop(['FUGA'],1).hist(figsize=(12, 8))
plt.show()


# In[ ]:


#Frecuencias según la variable SEXO
df_id['SEXO'].value_counts()


# In[ ]:


#Diagrama de pastel de la variable ESTADO CVIL
df_id['ESTADO_CIVIL'].value_counts().plot(kind='pie', autopct='%.2f', figsize=(6, 6))
plt.title('ESTADO CIVIL')
plt.ylabel('')


# In[ ]:


#Frecuencias según la variable ESTADO CIVIL
df_id['ESTADO_CIVIL'].value_counts()


# In[ ]:


#Diagrama de pastel de la variable SITUACION LABORAL
df_id['SITUACION_LABORAL'].value_counts().plot(kind='pie', autopct='%.2f', figsize=(6, 6))
plt.title('ESTADO CIVIL')
plt.ylabel('')


# In[ ]:


#Frecuencias según la variable SITUACION LABORAL
df_id['SITUACION_LABORAL'].value_counts()


# In[ ]:


#Tabla de contingencia entre FUGA y SEXO
df_sexo=pd.crosstab(index=df_id['FUGA'],
            columns=df_id['SEXO'], margins=True)
df_sexo


# In[ ]:


#Tabla de contingencia entre FUGA y ESTADO CIVIL
df_civil=pd.crosstab(index=df_id['FUGA'],
            columns=df_id['ESTADO_CIVIL'], margins=True)
df_civil


# In[ ]:


#Tabla de contingencia entre FUGA y SITUACION
df_cont=pd.crosstab(index=df_id['FUGA'],
            columns=df_id['SITUACION_LABORAL'], margins=True)
df_cont


# In[ ]:


#Tabla de contingencia entre FUGA y SEXO expresado como porcentaje relativo total
pd.crosstab(index=df_id['FUGA'],columns=df_id['SEXO'], margins=True).apply(lambda r: r/len(df_id) *100,axis=1)


# In[ ]:


#Tabla de contingencia entre FUGA y ESTADO CIVIL expresado como porcentaje relativo total
pd.crosstab(index=df_id['FUGA'],columns=df_id['ESTADO_CIVIL'], margins=True).apply(lambda r: r/len(df_id) *100,axis=1)


# In[ ]:


#Tabla de contingencia entre FUGA y SITUACION LABORAL expresado como porcentaje relativo total
pd.crosstab(index=df_id['FUGA'],columns=df_id['SITUACION_LABORAL'], margins=True).apply(lambda r: r/len(df_id) *100,axis=1)


# In[ ]:


#Tabla de contingencia en porcentaje relativo según sexo
pd.crosstab(index=df_id['FUGA'],columns=df_id['SEXO']).apply(lambda r: r/r.sum() *100,axis=1)


# In[ ]:


#Tabla de contingencia en porcentaje relativo según estado civil
pd.crosstab(index=df_id['FUGA'],columns=df_id['ESTADO_CIVIL']).apply(lambda r: r/r.sum() *100,axis=1)     


# In[ ]:


#Tabla de contingencia en porcentaje relativo según situacion laboral
pd.crosstab(index=df_id['FUGA'],columns=df_id['SITUACION_LABORAL']).apply(lambda r: r/r.sum() *100,axis=1)                                


# In[ ]:


# Gráfico de barras de fuga segun situación laboral
pd.crosstab(index=df_id['FUGA'],columns=df_id['SITUACION_LABORAL']
                  ).apply(lambda r: r/r.sum() *100,
                                              axis=1).plot(kind='bar',figsize=(10,10))


# In[ ]:


# Gráfico de barras de fuga segun estado civil
pd.crosstab(index=df_id['FUGA'],columns=df_id['ESTADO_CIVIL']
                  ).apply(lambda r: r/r.sum() *100,
                                              axis=1).plot(kind='bar',figsize=(10,10))


# In[ ]:


# Box plot de fuga segun tiempo alta
pd.crosstab(columns=df_id['FUGA'],index=df_id['TIEMPO_ALTA']
                  ).apply(lambda r: r/r.sum() *100,
                                              axis=1).plot(kind='box',figsize=(10,10))


# In[ ]:


# Box plot de fuga segun edad
pd.crosstab(columns=df_id['FUGA'],index=df_id['EDAD']
                  ).apply(lambda r: r/r.sum() *100,
                                              axis=1).plot(kind='box',figsize=(10,10))


# In[ ]:


#Histograma de fuga según la edad
pd.crosstab(columns=df_id['FUGA'],index=df_id['EDAD']
                  ).apply(lambda r: r/r.sum() *100,
                                              axis=1).plot(kind='hist',figsize=(10,10),alpha=0.7)


# In[ ]:


#Histograma de fuga según tiempo alta
pd.crosstab(columns=df_id['FUGA'],index=df_id['TIEMPO_ALTA']
                  ).apply(lambda r: r/r.sum() *100,
                                              axis=1).plot(kind='hist',figsize=(10,10),alpha=0.7)


# In[ ]:


# Box plot de fuga segun prom abono nomina
df_id.boxplot('PROM_TA_ABONO_NOMINA', by='FUGA', figsize=(12, 8))


# In[ ]:


# Box plot de fuga segun prom saldo ahorros
df_id.boxplot('PROM_TA_SALDO_AHORROS', by='FUGA', figsize=(12, 8))


# In[ ]:


# Box plot de fuga segun prom saldo creditos
df_id.boxplot('PROM_TA_SALDO_CREDITO', by='FUGA', figsize=(12, 8))


# <b>8. Evaluation</b>
# 
# Determinar la calidad del modelo realizado

# <b>ANOVA</b>

# In[ ]:


import statsmodels.api as sm
from statsmodels.formula.api import ols


# In[ ]:


mod = ols('EDAD ~ FUGA',
                data=df_id).fit()
                
aov_table = sm.stats.anova_lm(mod, typ=2)
print(aov_table)


# In[ ]:


mod = ols('TIEMPO_ALTA ~ FUGA',
                data=df_id).fit()
                
aov_table = sm.stats.anova_lm(mod, typ=2)
print(aov_table)


# In[ ]:


mod = ols('PROM_TA_SALDO_CREDITO ~ FUGA',
                data=df_id).fit()
                
aov_table = sm.stats.anova_lm(mod, typ=2)
print(aov_table)


# In[ ]:


mod = ols('PROM_TA_SALDO_AHORROS ~ FUGA',
                data=df_id).fit()
                
aov_table = sm.stats.anova_lm(mod, typ=2)
print(aov_table)


# In[ ]:


mod = ols('PROM_TA_ABONO_NOMINA ~ FUGA',
                data=df_id).fit()
                
aov_table = sm.stats.anova_lm(mod, typ=2)
print(aov_table)


# In[ ]:


df_id.describe()


# In[ ]:


df_true=df_id.query('FUGA==True')


# In[ ]:


df_true.mean()


# In[ ]:


df_false=df_id.query('FUGA==False')


# In[ ]:


df_false.mean()


# <b> ÁRBOL DE DECISIONES</b>

# In[ ]:


from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
import sklearn.metrics


# In[ ]:


#Directorio
os.chdir("C://Users/hecansaga/Desktop/YPD")


# In[ ]:


#Limpiar de datos missing
df_idc = df_id.dropna()


# In[ ]:


#Definir variables predictoras y objetivo
predictors = df_idc[['EDAD','TIEMPO_ALTA','PROM_TA_ABONO_NOMINA','PROM_TA_SALDO_AHORROS','PROM_TA_SALDO_CREDITO']]
targets = df_idc.FUGA


# In[ ]:


#Muestra de entrenamiento y de test con un test del 40%
pred_train, pred_test, tar_train, tar_test = train_test_split(predictors, targets, test_size=.2)


# In[ ]:


#tamaño muestra predictora train
pred_train.shape


# In[ ]:


#tamaño muestra predictora test
pred_test.shape


# In[ ]:


#tamaño muestra objetivo train
tar_train.shape


# In[ ]:


#tamaño muestra objetivo test
tar_test.shape


# In[ ]:


#Árbol
classifier=DecisionTreeClassifier()
classifier=classifier.fit(pred_train,tar_train)


# In[ ]:


#Predicciones
predictions=classifier.predict(pred_test)


# In[ ]:


#Matriz de confusión de predicciones
sklearn.metrics.confusion_matrix(tar_test,predictions)


# In[ ]:


#Indice de precision
sklearn.metrics.accuracy_score(tar_test, predictions)


# In[ ]:


#Librerías para exportar arbol
from sklearn import tree
from io import StringIO
from IPython.display import Image


# In[ ]:


#Exportar arbol
out = StringIO()
tree.export_graphviz(classifier, out_file='treeYPD.dot')

