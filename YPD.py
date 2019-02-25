#!/usr/bin/env python
# coding: utf-8

# <h1>PRUEBA PRÁCTICA</h1>
# <h3>YOUNG PROFESSIONAL DATA</h3>

# DESARROLLADO POR: 
# <b>Andrea Carolina Sánchez Valdés</b>

# In[163]:


import matplotlib
import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd
import seaborn as sns


# In[2]:


id_path=("C://Users/hecansaga/Desktop/YPD/BASE_ID.txt")
df_id = pd.read_table(id_path)
df_id.head()


# In[3]:


bm_path=("C://Users/hecansaga/Desktop/YPD/BASE_MOVIMIENTOS.txt")                   
df_bm = pd.read_fwf(bm_path)
df_bm.head()


# In[4]:


df_bm.dtypes


# In[5]:


df_bm.replace(regex=True,inplace=True,to_replace=r'\$ ',value=r'') 
df_bm.replace(regex=True,inplace=True,to_replace=r'\.',value=r'')
df_bm.replace(regex=True,inplace=True,to_replace=r'\,',value=r'.')


# In[6]:


df_bm['SALDO_FONDOS']=df_bm['SALDO_FONDOS'].astype("float64")
df_bm['SALDO_CREDITO1']=df_bm['SALDO_CREDITO1'].astype("float64")
df_bm['SALDO_CREDITO2']=df_bm['SALDO_CREDITO2'].astype("float64")
df_bm['SALDO_ACTIVO']=df_bm['SALDO_ACTIVO'].astype("float64")
df_bm['SALDO_PASIVO']=df_bm['SALDO_PASIVO'].astype("float64")


# In[7]:


df_bm.dtypes


# In[8]:


df_bm['FECHA_INFORMACION'].replace(regex=True,inplace=True,to_replace=r'\:00',value=r'')


# In[9]:


df_bm['FECHA_INFORMACION'].replace(regex=True,inplace=True,
                                   to_replace=['JAN','FEB','MAR','APR','MAY','JUN','JUL','AUG','SEP','OCT','NOV','DEC'],
                                   value=['/01/','/02/','/03/','/04/','/05/','/06/','/07/','/08/','/09/','/10/','/11/','/12/'])


# In[10]:


df_bm['FECHA_INFORMACION']=pd.to_datetime(df_bm['FECHA_INFORMACION'])
df_bm.dtypes


# In[11]:


df_bm.describe()


# In[12]:


df_bm.head()


# In[13]:


df_id.dtypes


# In[14]:


df_id.rename(columns={'fuga':'FUGA','CLIENTE_CC':'ID'},inplace=True)


# In[15]:


df_id.replace(regex=True,inplace=True,to_replace=r'\$ ',value=r'') 
df_id.replace(regex=True,inplace=True,to_replace=r'\.',value=r'')
df_id.replace(regex=True,inplace=True,to_replace=r'\,',value=r'.')


# In[16]:


df_id['FUGA']=df_id['FUGA'].fillna(0)
df_id['MES_DE_FUGA']=df_id['MES_DE_FUGA'].fillna(0)


# In[17]:


df_id['SEXO'].unique()


# In[18]:


df_id['SEXO'].replace(inplace=True,to_replace=['F', 'HOMBRE', 'M', 'Hombre', 'mujer', 'femenino', 'masculino',
       'FEMENINO', 'Mujer', 'varón', 'Masc', 'MUJER'],
                      value=['FEMENINO', 'MASCULINO', 'MASCULINO', 'MASCULINO', 'FEMENINO', 'FEMENINO', 'MASCULINO',
       'FEMENINO', 'FEMENINO', 'MASCULINO', 'MASCULINO', 'FEMENINO'])


# In[19]:


df_id['ESTADO_CIVIL'].unique()


# In[20]:


df_id['SITUACION_LABORAL'].unique()


# In[21]:


df_id['SITUACION_LABORAL'].replace(inplace=True,to_replace=['otros', 'Contrato fijo', 'OTROS', 'contrato autonomo',
       ' desconocido   ', 'CONTRATO AUTONOMO', 'CONTRATO FIJO', 'CONTRATO TEMPORAL', 'temporal     ', 'SIN CLASIFICAR'],
                                   value=['OTROS', 'CONTRATO FIJO', 'OTROS', 'CONTRATO AUTONOMO',
       'DESCONOCIDO', 'CONTRATO AUTONOMO', 'CONTRATO FIJO', 'CONTRATO TEMPORAL', 'CONTRATO TEMPORAL', 'SIN CLASIFICAR'])


# In[22]:


df_id['FECHA_ALTA'].replace(regex=True,inplace=True,
                            to_replace=['ene','feb','mar','abr','may','jun','jul','ago','sep','oct','nov','dic'],
                            value=['01','02','03','04','05','06','07','08','09','10','11','12'])


# In[23]:


df_mesid1=df_id['FECHA_ALTA'].str[:2].astype(object)
df_mesid1.head()


# In[24]:


df_añoid1=df_id['FECHA_ALTA'].str[4:].astype(object)
df_añoid1.head()


# In[25]:


df_diaid1=df_id['FECHA_ALTA'].str[2:4:].astype(object)
df_diaid1.head()


# In[26]:


df_mesid2=df_id['FECHA_NACIMIENTO'].str[4:6:].astype(object)
df_mesid2.head()


# In[27]:


df_añoid2=df_id['FECHA_NACIMIENTO'].str[:4].astype(object)
df_añoid2.head()


# In[28]:


df_diaid2=df_id['FECHA_NACIMIENTO'].str[6:].astype(object)
df_diaid2.head()


# In[29]:


df_id['FECHA_ALTA'] = df_diaid1+'/'+df_mesid1+'/'+df_añoid1
df_id['FECHA_ALTA'].head()


# In[30]:


df_id['FECHA_NACIMIENTO'] = df_diaid2+'/'+df_mesid2+'/'+df_añoid2
df_id['FECHA_NACIMIENTO'].head()


# In[31]:


df_len=df_id['FECHA_NACIMIENTO'].str.len()
df_len.head()


# In[32]:


df_len.idxmax()


# In[33]:


df_id['FECHA_NACIMIENTO'].replace(inplace=True,to_replace='1-01/-0/0001',value='NaN')


# In[34]:


df_id['ID']=df_id['ID'].astype('float64')
df_id['ID']=df_id['ID'].astype('int64')
df_id['FECHA_ALTA']=pd.to_datetime(df_id['FECHA_ALTA'])
df_id['FECHA_NACIMIENTO']=pd.to_datetime(df_id['FECHA_NACIMIENTO'])
df_id['FUGA']=df_id['FUGA'].astype('bool')
df_id['MES_DE_FUGA']=df_id['MES_DE_FUGA'].astype('int64')
df_id.dtypes


# In[35]:


df_bm=df_bm.sort_values(by=['ID','FECHA_INFORMACION'])
df_bm.head()


# In[36]:


df_join=pd.merge(df_bm,df_id[['ID','FUGA','MES_DE_FUGA']],on='ID',how='left')
df_join.head()


# In[37]:


df_join.tail()


# In[38]:


#Promedio abono nomina

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


# In[39]:


ids=df_join.groupby('ID').size()

for i in range(2,2501):      
    countid=ids.at[i,]
    j=np.where(df_join["ID"] == i)[0]
    k=j[0]
    df_join.at[k,'PROM_TRIM_ABONOS_NOMINA']="NaN"
    df_join.at[k+1,'PROM_TRIM_ABONOS_NOMINA']="Nan"
    i=i+1
    if i>2501:
        break


# In[40]:


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


# In[41]:


ids=df_join.groupby('ID').size()

for i in range(2,2501):      
    countid=ids.at[i,]
    j=np.where(df_join["ID"] == i)[0]
    k=j[0]
    df_join.at[k,'PROM_TRIM_SALDO_AHORROS']="NaN"
    df_join.at[k+1,'PROM_TRIM_SALDO_AHORROS']="Nan"
    i=i+1
    if i>2501:
        break


# In[42]:


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


# In[43]:


ids=df_join.groupby('ID').size()

for i in range(2,2501):      
    countid=ids.at[i,]
    j=np.where(df_join["ID"] == i)[0]
    k=j[0]
    df_join.at[k,'PROM_TRIM_SALDO_CREDITO']="NaN"
    df_join.at[k+1,'PROM_TRIM_SALDO_CREDITO']="Nan"
    i=i+1
    if i>2501:
        break


# In[44]:


df_join.head()


# In[110]:


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


# In[146]:


df_id.head(100)


# In[114]:


df_id.corr()


# In[164]:


sns.heatmap(df_id.corr(), square=True, annot=True)


# In[130]:


num_bins=6
n,bins,patches = plt.hist(df_id['MES_DE_FUGA'],num_bins, alpha = 0.5 ) 
plt.title('Histograma mes fuga')
plt.xlabel('Mes')
plt.ylabel('Frencuencia')
plt.show()


# In[153]:


fig = plt.figure()

ax0 = fig.add_subplot(2, 2, 1) 
ax1 = fig.add_subplot(2, 2, 2) 
ax2 = fig.add_subplot(2, 2, 3)

df_id['PROM_TA_SALDO_CREDITO'].plot(kind='box', figsize=(10, 15), ax=ax0)
ax0.set_title('Box plot saldo credito promedio trimestre')
ax0.set_ylabel('Promedio saldo credito')

df_id['PROM_TA_SALDO_AHORROS'].plot(kind='box', figsize=(10, 15), ax=ax1)
ax1.set_title ('Box plot saldo ahorros promedio trimestre')
ax1.set_ylabel('Promedio saldo ahorros')

df_id['PROM_TA_ABONO_NOMINA'].plot(kind='box', figsize=(10, 15), ax=ax2)
ax2.set_title ('Box plot Abono nomina promedio trimestral')
ax2.set_ylabel('Abono nomina promedio')

plt.show()


# In[168]:


df_id['SEXO'].value_counts()


# In[185]:


df_id['ESTADO_CIVIL'].value_counts().plot(kind='pie', autopct='%.2f', figsize=(6, 6))
plt.title('ESTADO CIVIL')
plt.ylabel('')


# In[179]:


df_id['ESTADO_CIVIL'].value_counts()


# In[184]:


df_id['SITUACION_LABORAL'].value_counts().plot(kind='pie', autopct='%.2f', figsize=(6, 6))
plt.title('ESTADO CIVIL')
plt.ylabel('')


# In[178]:


df_id['SITUACION_LABORAL'].value_counts()


# In[188]:


df_sexo=pd.crosstab(index=df_id['FUGA'],
            columns=df_id['SEXO'], margins=True)
df_sexo


# In[189]:


df_civil=pd.crosstab(index=df_id['FUGA'],
            columns=df_id['ESTADO_CIVIL'], margins=True)
df_civil


# In[190]:


df_cont=pd.crosstab(index=df_id['FUGA'],
            columns=df_id['SITUACION_LABORAL'], margins=True)
df_cont


# In[191]:


pd.crosstab(index=df_id['FUGA'],columns=df_id['SEXO'], margins=True).apply(lambda r: r/len(df_id) *100,axis=1)


# In[192]:


pd.crosstab(index=df_id['FUGA'],columns=df_id['ESTADO_CIVIL'], margins=True).apply(lambda r: r/len(df_id) *100,axis=1)


# In[193]:


pd.crosstab(index=df_id['FUGA'],columns=df_id['SITUACION_LABORAL'], margins=True).apply(lambda r: r/len(df_id) *100,axis=1)


# In[197]:


pd.crosstab(index=df_id['FUGA'],columns=df_id['ESTADO_CIVIL']).apply(lambda r: r/r.sum() *100,axis=1)     


# In[195]:


pd.crosstab(index=df_id['FUGA'],columns=df_id['SITUACION_LABORAL']).apply(lambda r: r/r.sum() *100,axis=1)                                


# In[198]:


pd.crosstab(index=df_id['FUGA'],columns=df_id['SEXO']).apply(lambda r: r/r.sum() *100,axis=1)


# In[206]:


pd.crosstab(index=df_id['FUGA'],columns=df_id['SITUACION_LABORAL']
                  ).apply(lambda r: r/r.sum() *100,
                                              axis=1).plot(kind='bar',figsize=(10,10))


# In[207]:


pd.crosstab(index=df_id['FUGA'],columns=df_id['ESTADO_CIVIL']
                  ).apply(lambda r: r/r.sum() *100,
                                              axis=1).plot(kind='bar',figsize=(10,10))


# In[218]:


df_id['PROM_TA_ABONO_NOMINA'].plot(kind='area', figsize=(10,10))


# In[219]:


df_id['PROM_TA_SALDO_AHORROS'].plot(kind='area',figsize=(10,10))


# In[220]:


df_id['PROM_TA_SALDO_CREDITO'].plot(kind='area',figsize=(10,10))


# In[ ]:




