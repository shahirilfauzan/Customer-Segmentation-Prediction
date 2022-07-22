# -*- coding: utf-8 -*-
"""
Created on Fri Jul 22 09:23:36 2022

@author: Shah
"""

from customer_segmentation_module import EDA,ModelEvaluation,ModelDevelopment
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras.callbacks import EarlyStopping,TensorBoard
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import plot_model
from sklearn.impute import KNNImputer

import matplotlib.pyplot as plt
import missingno as msno
import pandas as pd
import numpy as np
import datetime
import pickle
import os

#%% Constant
CSV_PATH=os.path.join(os.getcwd(),'dataset','Train.csv')
MMS_PATH=os.path.join(os.getcwd(),'model','mms.pkl')
OHE_PATH=os.path.join(os.getcwd(), 'model','ohe_encoder.pkl')
MODEL_SAVE_PATH=os.path.join(os.getcwd(),'model','model.h5')
#%% Step 1) Data Loading
df = pd.read_csv(CSV_PATH)
#%% Step 2) Data Inspection
df.info()
df=df.drop(labels='id',axis=1)

cat=list(df.columns[df.dtypes == 'object'])
cat.append('day_of_month')
cat.append('term_deposit_subscribed')

con=list(df.columns[(df.dtypes=='int64') | (df.dtypes=='float64')])
con.remove('day_of_month')
con.remove('term_deposit_subscribed')

df.groupby(['term_deposit_subscribed','job_type']).agg({'term_deposit_subscribed':'count'}).plot(kind='bar')
df.groupby(['term_deposit_subscribed','education']).agg({'term_deposit_subscribed':'count'}).plot(kind='bar')
df.groupby(['term_deposit_subscribed','marital']).agg({'term_deposit_subscribed':'count'}).plot(kind='bar')
df.groupby(['term_deposit_subscribed','housing_loan']).agg({'term_deposit_subscribed':'count'}).plot(kind='bar')

eda = EDA()
eda.displot_graph(con,df)
eda.countplot_graph(cat,df)

#%% Step 3) Data Cleaning
# Label Encoder

for i in cat: 
    if i == 'term_deposit_subscribed': 
        continue
    else: 
           le = LabelEncoder()
           temp = df[i]
           temp[temp.notnull()]=le.fit_transform(temp[df[i].notnull()])
           df[i]= pd.to_numeric(df[i], errors = 'coerce')
           PICKLE_SAVE_PATH = os.path.join(os.getcwd(),'model',i+'encoder.pkl')
           with open(PICKLE_SAVE_PATH, 'wb') as file: 
               pickle.dump(le,file)
i + '.pkl'

df.info()
df.isna().sum()

#MSNO BAR
msno.matrix(df)
msno.bar(df)

#too many NaNs inside days_since_prev_campaign_contact, decided to drop it

df=df.drop(labels='days_since_prev_campaign_contact',axis=1)

#Remove NaNs
#KNN imputer
columns_name=df.columns

knn_i=KNNImputer()
df=knn_i.fit_transform(df) #return numpy array
df=pd.DataFrame(df) # to convert back into dataframe
df.columns = columns_name

df.info()
df.isna().sum()
temp = df.describe().T

df['customer_age'] = np.floor(df['customer_age'])
df['personal_loan'] = np.floor(df['personal_loan'])
df['last_contact_duration'] = np.floor(df['last_contact_duration'])

temp=df.describe().T

# 3) Remove Duplicate
df.duplicated().sum()# no duplicate
#%% Step 4) Features Selection

X=df.drop(labels='term_deposit_subscribed',axis=1)
y=df['term_deposit_subscribed'].astype(int)

con=['customer_age','balance','last_contact_duration',
     'num_contacts_in_campaign','num_contacts_prev_campaign']

# CON VS CAT
#LINEAR REGRESSION

selected_features=[]

for i in con:
    lr = LogisticRegression()
    lr.fit(np.expand_dims(X[i],axis=1),y)
    print(i)
    print(lr.score(np.expand_dims(X[i],axis=-1),y))
    if lr.score(np.expand_dims(X[i],axis=-1),y) > 0.8:
        selected_features.append(i)
    
print(selected_features)

#Categorical vs categorical
#cramers'v

for i in cat:
    print(i)
    matrix = pd.crosstab(df[i],y).to_numpy()
    print(EDA().cramers_corrected_stat(matrix))
    if EDA().cramers_corrected_stat(matrix) > 0.3:
        selected_features.append(i)
        
print(selected_features)
#From above analysis only from continuous features and prev_campaign_outcome
#from categorical are selected

df=df.loc[:,selected_features] # to c heck the list of selected features
X=df.drop(labels='term_deposit_subscribed',axis=1)
y=df['term_deposit_subscribed'].astype(int)

#%% Step 5) Data Preprocessing
#MMS
mms=MinMaxScaler()
X=mms.fit_transform(X)

with open(MMS_PATH,'wb')as file:
    pickle.dump(mms,file)

#OHE
ohe=OneHotEncoder(sparse=False)
y=ohe.fit_transform(np.expand_dims(y,axis=-1))

with open(OHE_PATH,'wb') as file:
    pickle.dump(ohe,file)
    
#%%
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,
                                                random_state=123)
#%% Model Development
input_shape=np.shape(X_train)[1:]
nb_class=len(np.unique(y,axis=0))

md=ModelDevelopment()
model=md.simple_dl_model(input_shape,nb_class,nb_node=128,dropout_rate=0.3)

plot_model(model,show_shapes=(True))
#%%

LOGS_PATH=os.path.join(os.getcwd(),'logs',datetime.datetime.now().
                       strftime('%Y%m%d-%H%M%S'))

tensorboard_callback=TensorBoard(log_dir=LOGS_PATH,histogram_freq=1)
early_callback=EarlyStopping(monitor='val_acc',patience=3)

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['acc'])

hist=model.fit(X_train,y_train,epochs=50,
               validation_data=(X_test,y_test),
               callbacks=[early_callback,tensorboard_callback])

#%%model evaluation
print(hist.history.keys())
key=list(hist.history.keys())

#plot_hist_graph
me=ModelEvaluation()
plot_hist=me.plot_hist_graph(hist,key,a=0,b=2,c=1,d=3)
#%%model analysis
y_pred=np.argmax(model.predict(X_test),axis=1)
y_test=np.argmax(y_test,axis=1)
print(classification_report(y_test,y_pred))
print(confusion_matrix(y_test,y_pred))

cm=confusion_matrix(y_test,y_pred)

labels=['not subscribed','subscribed']
disp=ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=labels)
disp.plot(cmap=plt.cm.Blues)
plt.show()

model.save(MODEL_SAVE_PATH)