""" Writing my first randomforest code.
Author : Yuzuru Kato
Date : 12rd Jan 2015
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

from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn import linear_model

from sklearn import decomposition
from sklearn import grid_search


def PreProcessing(train_df):
    """pre-processes the train dataframe.
    pram train_df:train dataframe.
    """
    # I need to convert all strings to integer classifiers.
    # I need to fill in the missing values of the data and make it complete.

    # female = 0, Male = 1
    train_df['Gender'] = train_df['Sex'].map( {'female': 0, 'male': 1} ).astype(int)

    # Embarked from 'C', 'Q', 'S'
    # Note this is not ideal: in translating categories to numbers, Port "2" is not 2 times greater than Port "1", etc.

    # All missing Embarked -> just make them embark from most common place
    if len(train_df.Embarked[ train_df.Embarked.isnull() ]) > 0:
        train_df.Embarked[ train_df.Embarked.isnull() ] = train_df.Embarked.dropna().mode().values

    Ports = list(enumerate(np.unique(train_df['Embarked'])))    # determine all values of Embarked,
    Ports_dict = { name : i for i, name in Ports }              # set up a dictionary in the form  Ports : index
    train_df.Embarked = train_df.Embarked.map( lambda x: Ports_dict[x]).astype(int)     # Convert all Embark strings to int

    # All the ages with no data -> make the median of all Ages
    median_age = train_df['Age'].dropna().median()
    if len(train_df.Age[ train_df.Age.isnull() ]) > 0:
        train_df.loc[ (train_df.Age.isnull()), 'Age'] = median_age
    return train_df

def PCA_func(X_data, pca_num):
    """Performes PCA with pca_num's components to X_data.
    param X_data: data to which PCA is performed
    param pca_num: number of components to keep
    """
    pca = decomposition.PCA()
    pca.n_components = pca_num
    X_data = pca.fit_transform(X_data)
    print("PCA_performed")  
    print(pca.explained_variance_)
    #print(pca.components_)
    return X_data, pca.components_
    
def evaluation_func(true_surv,predicted_surv,TF_list):
    """Ouputs the results of several evaluation schemes.
    :param true_surv: true surviveds for test passengers.
    :param predicted_surv: predicted surviveds for test passengers.
    :Flag lists for whether it calculates the output for each evaluation:
    TF_list[0]:Confusion Matrix
    TF_list[1]:Accuracy
    TF_list[2]:P,R,F,S values
    TF_list[3]:Precision-recall curve
    TF_list[4]:ROC curve
    """
    #Prints Confusion Matrix
    if(TF_list[0] == True):
        print(confusion_matrix(true_surv,predicted_surv))

    #Prints Accuracy
    if(TF_list[1] == True):
        print(accuracy_score(true_surv,predicted_surv))

    ##Prints P,R,F,S values 
    if(TF_list[2] == True):
        target_names = ['Dead', 'Alive'];
        print(classification_report(true_surv, predicted_surv, target_names=target_names))

    #plots precision-recall curve
    if(TF_list[3] == True):
        precision, recall, pr_thresholds = precision_recall_curve(true_surv,predicted_surv)
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
        fpr, tpr, roc_thresholds = roc_curve(true_surv,predicted_surv)
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

    if(TF_list[3] == True | TF_list[4] == True):
        plt.show()

def get_Kfold_CrossValidationScore(X_data,Y_data, est, Fold_num):
    """ Outputs CrossValidationScore
    param: Input data
    param: Output data
    est: Estimatior
    Fold: the Value of K
    """
    #Prints cross_validation score
    X_folds = np.array_split(X_data, Fold_num)
    y_folds = np.array_split(Y_data, Fold_num)
    scores = list()
    for k in range(Fold_num):
         # We use 'list' to copy, in order to 'pop' later on
         X_train = list(X_folds)
         X_test  = X_train.pop(k)
         X_train = np.concatenate(X_train)
         y_train = list(y_folds)
         y_test  = y_train.pop(k)
         y_train = np.concatenate(y_train)
         scores.append(est.fit(X_train, y_train).score(X_test, y_test))
    print(scores)
    
# Set parameters
T_size = 100; #Number Of Test data Size

#NAMEs for Estimator
EST_RF = 0
EST_SVC = 1
EST_LOGREG =2

EST = EST_RF    #Define Estimatior

PCA = False            #Whether or not PCA is performed 
pca_num = 7

GSCV = True          #Whether or not GridSearchCV is performed (currently only for SVC)

# Data cleanup
# TRAIN DATA
train_df = pd.read_csv('train.csv', header=0)        # Load the train file into a dataframe

#Pre-processes the data frame
train_df = PreProcessing(train_df)

# Collect the train_srvs, test(true)_srvs,ids

train_srvs = train_df['Survived'].values[:-T_size]
test_srvs = train_df['Survived'].values[-T_size:]
ids = train_df['PassengerId'].values[-T_size:]

# Remove the Name column, Cabin, Ticket, and Sex (since I copied and filled it to Gender), PassengerId, and Survived
train_df = train_df.drop(['Name', 'Sex', 'Ticket', 'Cabin','PassengerId','Survived'], axis=1) 

# Get TRAIN DATA
train_data = train_df.values[:-T_size,:]
if(PCA == True):
    train_data, pca_components = PCA_func(train_data,pca_num)

# Get TEST DATA
test_data = train_df.values[-T_size:,:]
if(PCA == True):
    test_data = test_data.dot(pca_components.T)



#The data is now ready to go. So lets fit to the train, then predict to the test!
#Convert back to a numpy array

if(EST == EST_RF):
    print ('Random Forst Training...')
    est = RandomForestClassifier(n_estimators=100)
    est = est.fit(train_data, train_srvs)
elif(EST == EST_SVC):
    print ('SVC Training...')
    if(GSCV == True):
        print ('With GridSearchCV...')
        parameters = {'kernel':('rbf','sigmoid'), 'C':[1, 10], \
        'gamma':[  1.00000000e-06,   3.59381366e-06,   1.29154967e-05, \
        4.64158883e-05,   1.66810054e-04,   5.99484250e-04, \
        2.15443469e-03,   7.74263683e-03,   2.78255940e-02, \
        1.00000000e-01]} 
        #parameters = {'kernel':('linear', 'rbf','poly','sigmoid')}
        #parameters = {'kernel':('rbf','sigmoid')}
        svc = svm.SVC()
        est = grid_search.GridSearchCV(svc, parameters)
        est.fit(train_data, train_srvs)

        print("Best parameters set found on development set:")
        print()
        print(est.best_estimator_)
    else:
        knl = 'rbf'
        est = svm.SVC(kernel=knl)
        print('kernel is' , knl)
    est = est.fit(train_data, train_srvs)
elif(EST == EST_LOGREG):
    print ('Logistic Regression Training...')
    est = linear_model.LogisticRegression(penalty = 'l2')
    est = est.fit(train_data, train_srvs)


print ('Predicting...')
predicted_srvs = est.predict(test_data).astype(int)

predictions_file = open("EstimationResult.csv", "w",newline='')
open_file_object = csv.writer(predictions_file)
open_file_object.writerow(["PassengerId","PredictedSurvived","TrueSurvived"])
open_file_object.writerows(zip(ids,predicted_srvs,test_srvs))
predictions_file.close()
print ('T_size is %s' % len(test_srvs))

#evaluates prediction
evaluation_func(test_srvs,predicted_srvs,[True,True,True,False,False])

#evaluates crossValidation Score
get_Kfold_CrossValidationScore(train_data, train_srvs,est,3)






