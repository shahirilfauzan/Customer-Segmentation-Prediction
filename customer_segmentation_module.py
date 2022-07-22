# -*- coding: utf-8 -*-
"""
Created on Fri Jul 15 19:26:34 2022

@author: Shah
"""


from tensorflow.keras.layers import Dense,Dropout,BatchNormalization
from tensorflow.keras import Input,Sequential 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as ss

class EDA:
    def displot_graph(self,con,df):
        '''
        distplot graph for continuous data

        Parameters
        ----------
        con : TYPE
            DESCRIPTION.
        df : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        '''
        for i in con:
            plt.figure()
            sns.distplot(df[i])
            plt.show()
    
    def countplot_graph(self,cat,df):
        '''
        countplot graph for categorical data

        Parameters
        ----------
        cat : TYPE
            DESCRIPTION.
        df : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        '''
        for i in cat:
            plt.figure()
            sns.countplot(df[i])
            plt.show()
            
    def cramers_corrected_stat(self,confusion_matrix):
        """ calculate Cramers V statistic for categorial-categorial association.
            uses correction from Bergsma and Wicher,
            Journal of the Korean Statistical Society 42 (2013): 323-328
        """
        chi2 = ss.chi2_contingency(confusion_matrix)[0]
        n = confusion_matrix.sum()
        phi2 = chi2/n
        r,k = confusion_matrix.shape
        phi2corr = max(0, phi2 - ((k-1)*(r-1))/(n-1))  
        rcorr = r - ((r-1)**2)/(n-1)
        kcorr = k - ((k-1)**2)/(n-1)
        return np.sqrt(phi2corr / min( (kcorr-1), (rcorr-1)))


class ModelDevelopment:
    def simple_dl_model(self,input_shape,nb_class,nb_node=128,dropout_rate=0.3):
        '''
        This is a simple 3 layers deep learning model . it does not perform well if the dense is lesser 

        Parameters
        ----------
        X_train : TYPE
            DESCRIPTION.
        nb_class : TYPE
            DESCRIPTION.
        nb_node : TYPE, optional
            DESCRIPTION. The default is 128.
        dropout_rate : TYPE, optional
            DESCRIPTION. The default is 0.3.

        Returns
        -------
        model : TYPE
            DESCRIPTION.

        '''
        model=Sequential()
        model.add(Input(shape=input_shape))
        model.add(Dense(nb_node,activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(dropout_rate))
        model.add(Dense(nb_node,activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(dropout_rate))
        model.add(Dense(nb_node,activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(dropout_rate))
        model.add(Dense(nb_class,activation='softmax'))
        model.summary()
        
        return model


class ModelEvaluation:
    def plot_hist_graph(self,hist,key,a=0,b=2,c=1,d=3):
        plt.figure()
        plt.plot(hist.history[key[a]])
        plt.plot(hist.history[key[b]])
        plt.legend(['training_'+ str(key[a]), key[b]])
        plt.show()
        
        plt.figure()
        plt.plot(hist.history[key[c]])
        plt.plot(hist.history[key[d]])
        plt.legend(['training_'+ str(key[c]), key[d]])
        plt.show()
        