""" Writing my first randomforest code.
Author : Yuzuru Kato
Date : 18th Jan 2015
""" 
import pandas as pd
import numpy as np
import csv as csv

import matplotlib.pyplot as plt

from sklearn.metrics import precision_recall_curve
from sklearn.metrics import auc
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve, auc
from sklearn import cross_validation

from sklearn import decomposition

def pre_processing(train_df):
    """pre-processes(converts objects into numerical values) the train dataframe.
    Currently, pro-processing is performed to data which have dtypes other than 'int64';
    but it is supposed that data which has 'int 32', 'float 32', 'float 64' also not be preprocessed.
    param train_df:train dataframe.
    """
    for i in range(0,len(list(train_df))):
        key =list(train_df)[i] 
        if(train_df[key].dtypes == 'int64'):
            TY_INT = True
        else:
            TY_INT = False

        #print("{0}:{1}".format(Key,TY_INT))
        #Converts string to int
        if(TY_INT == False):
            Ports = list(enumerate(np.unique(train_df[key])))
            Ports_dict = { name : i for i, name in Ports }
            train_df[key] = train_df[key].map( lambda x: Ports_dict[x]).astype(int)       

    return train_df

def drop_unused_variables(train_df, used_variables_list):
    """Drop unused variable
    param train_df: train dataframe
    param used_variable_list: list of variables which we use
    """
    length = len(list(train_df))
    dr_idx = 0       
    for i in range(0,length):
        DROP = True
        for j in range(0,len(used_variables_list)):
            if(list(train_df)[dr_idx] == used_variables_list[j]):
                DROP = False
                
        if(DROP == False):
            #print("{0} is undrop".format(list(train_df)[dr_idx]))
            dr_idx += 1 #UnDrop used variables
        elif(DROP == True):
            #print("{0} is drop".format(list(train_df)[dr_idx]))
            train_df.drop(list(train_df)[dr_idx],axis = 1, inplace = True) #Drop unused variables
    return train_df

def get_pca(train_X_data, pca_num):
    """Performes PCA with pca_num's components to X_data.
    param X_data: data to which PCA is performed
    param pca_num: number of components to keep
    """
    pca = decomposition.PCA()
    pca.n_components = pca_num
    train_X_data = pca.fit_transform(train_X_data)
    print("PCA_performed")  
    print(pca.explained_variance_)
    #print(pca.components_)
    return train_X_data, pca.components_

def get_evaluation(test_Y_data,prd_Y_data,TF_list):
    """Ouputs the results of several evaluation schemes.
    :param test_Y_data: true surviveds for test passengers.
    :param prd_Y_data: predicted surviveds for test passengers.
    :Flag lists for whether it calculates the output for each evaluation:
    TF_list[0]:Confusion Matrix
    TF_list[1]:Accuracy
    TF_list[2]:P,R,F,S values
    TF_list[3]:Precision-recall curve
    TF_list[4]:ROC curve
    """
    #Prints Confusion Matrix
    if(TF_list[0] == True):
        print(confusion_matrix(test_Y_data,prd_Y_data))

    #Prints Accuracy
    if(TF_list[1] == True):
        print(accuracy_score(test_Y_data,prd_Y_data))

    ##Prints P,R,F,S values 
    if(TF_list[2] == True):
        target_names = ['Dead', 'Alive'];
        print(classification_report(test_Y_data,prd_Y_data, target_names=target_names))

    #plots precision-recall curve
    if(TF_list[3] == True):
        precision, recall, pr_thresholds = precision_recall_curve(test_Y_data,prd_Y_data)
        area = auc(recall, precision)
        print ("Area Under Curve: %0.2f" % area)
        plt.figure(0)
        plt.clf()
        plt.plot(recall, precision, label='Precision-Recall curve')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.ylim([0.0, 1.05])
        plt.xlim([0.0, 1.0])
        plt.title('Precision-Recall example: AUC=%0.2f' % area)
        plt.legend(loc="lower left")
        
    #Plots ROC curve
    if(TF_list[4] == True):
        fpr, tpr, roc_thresholds = roc_curve(test_Y_data,prd_Y_data)
        roc_auc = auc(fpr, tpr)
        print ("Area under the ROC curve : %f" % roc_auc)
        plt.figure(1)
        plt.clf()
        plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.0])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic example')
        plt.legend(loc="lower right")

    if(TF_list[3] == True or TF_list[4] == True):
        plt.show()

def get_Kfold_CrossValidation_score(train_X_data,train_Y_data, est, fold_num):
    """ Outputs CrossValidationScore
    param train_X_data: train_X_data
    param train_Y_data: train_Y_data
    est: Estimatior
    fold: the Value of K
    """
    #Prints cross_validation score
    X_folds = np.array_split(train_X_data, fold_num)
    y_folds = np.array_split(train_Y_data, fold_num)
    scores = list()
    for k in range(fold_num):
         # We use 'list' to copy, in order to 'pop' later on
         X_train = list(X_folds)
         X_test  = X_train.pop(k)
         X_train = np.concatenate(X_train)
         y_train = list(y_folds)
         y_test  = y_train.pop(k)
         y_train = np.concatenate(y_train)
         scores.append(est.fit(X_train, y_train).score(X_test, y_test))
    print(scores)

def plot_svc_result(train_X_data, test_X_data,svc):
    """ Outputs SVC results with 2 PCA components
    param X_train: X_train_data
    param X_test: X_test_data
    svc: Svc
    """ 
    plt.figure(0)
    plt.clf()
    plt.scatter(train_X_data[:, 0], train_X_data[:, 1], zorder=10, cmap=plt.cm.Paired)
    
    # Circle out the test data
    plt.scatter(test_X_data[:, 0], test_X_data[:, 1], s=80, facecolors='none', zorder=10)

    plt.axis('tight')
    x_min = train_X_data[:, 0].min()
    x_max = train_X_data[:, 0].max()
    y_min = train_X_data[:, 1].min()
    y_max = train_X_data[:, 1].max()

    XX, YY = np.mgrid[x_min:x_max:200j, y_min:y_max:200j]
    Z = svc.decision_function(np.c_[XX.ravel(), YY.ravel()])

    # Put the result into a color plot
    Z = Z.reshape(XX.shape)
    plt.pcolormesh(XX, YY, Z > 0, cmap=plt.cm.Paired)
    plt.contour(XX, YY, Z, colors=['k', 'k', 'k'], linestyles=['--', '-', '--'],
                levels=[-.5, 0, .5])

    plt.title('SCV_result')
    plt.show()
   
    
    


\
