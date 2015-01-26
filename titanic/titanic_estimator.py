""" Writing my first randomforest code.
Author : Yuzuru Kato
Date : 25th Jan 2015
"""

# Third party library
import pandas as pd
import numpy as np
import csv as csv
# import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn import linear_model
from sklearn import grid_search
import json

# My library
from tools import *

# Neilsen's library
import network2
from network_plot import *


def pre_processing_for_titanic(train_df):
    """pre-processes the train dataframe.
    pram train_df:train dataframe.
    """
    # I need to convert all strings to integer classifiers.
    # I need to fill in the missing values of the data and make it complete.

    # female = 0, Male = 1
    train_df['Gender'] = train_df['Sex'].map(
        {'female': 0, 'male': 1}).astype(int)

    # Embarked from 'C', 'Q', 'S'
    # Note this is not ideal: in translating categories to numbers, Port "2"
    # is not 2 times greater than Port "1", etc.

    # All missing Embarked -> just make them embark from most common place
    if len(train_df.Embarked[train_df.Embarked.isnull()]) > 0:
        train_df.Embarked[
            train_df.Embarked.isnull()] = \
            train_df.Embarked.dropna().mode().values

    # determine all values of Embarked,
    Ports = list(enumerate(np.unique(train_df['Embarked'])))
    # set up a dictionary in the form  Ports : index
    Ports_dict = {name: i for i, name in Ports}
    train_df.Embarked = train_df.Embarked.map(lambda x: Ports_dict[x]).astype(
        int)     # Convert all Embark strings to int

    # All the ages with no data -> make the median of all Ages
    median_age = train_df['Age'].dropna().median()
    if len(train_df.Age[train_df.Age.isnull()]) > 0:
        train_df.loc[(train_df.Age.isnull()), 'Age'] = median_age
    return train_df

# ------------
# Define Parameters

# Set parameters
CUT_TEST_NUM = 100  # Number Of test data Size

# NAMEs for Estimator
EST_RF = 0
EST_SVC = 1
EST_LOGREG = 2
EST_NN = 3

EST = EST_RF  # Define Estimatior

# For RF,SVC,LOGREG:Whether or not GridSearchCV is performed (currently
# only for SVC)
GSCV = False

# Whether or not PCA or LDA is performed
DIMRED_NONE = 0
DIMRED_PCA = 1
DIMRED_LDA = 2
DIMRED = DIMRED_NONE

pca_num = 2
lda_num = 5

# Wheather daata is impblanced or not(performs SMOTE or not)
IMBALANCE = False

# ------------
# Reads DATA
# Load the train file into a dataframe
train_df = pd.read_csv('train.csv', header=0)

# ------------
# Pre-Processing

# Pre-processes the data frame
train_df = pre_processing_for_titanic(train_df)

# Collect the train_srvs, test(true)_srvs,ids
train_Y_data = train_df['Survived'].values[:-CUT_TEST_NUM]
test_Y_data = train_df['Survived'].values[-CUT_TEST_NUM:]
ids = train_df['PassengerId'].values[-CUT_TEST_NUM:]

# Remove the Name column, Cabin, Ticket, and Sex (since I copied and
# filled it to Gender), PassengerId, and Survived
train_df = train_df.drop(
    ['Name', 'Sex', 'Ticket', 'Cabin', 'PassengerId', 'Survived'], axis=1)

# Left only two parameters(Pclass,Age,SibSp,Parch,Fare,Embarked,Gender)
# #train_df = train_df.drop(['Pclass', 'SibSp',
# #             'Parch', 'Gender','Embarked'], axis=1)

# Get TRAIN DATA and TEST DATA
train_X_data = train_df.values[:-CUT_TEST_NUM, :]
test_X_data = train_df.values[-CUT_TEST_NUM:, :]

# Performs over_sampling and under_sampling
if IMBALANCE:
    O_N = 200
    O_k = 4
    U_N = 80
    train_X_data, train_Y_data = get_smote(
        train_X_data, train_Y_data, O_N, O_k, U_N)

# For Debug
# #print(len(train_srvs[train_srvs == 0]))
# #print(len(train_srvs[train_srvs == 1]))

# Performs dimention reduction
if(DIMRED == DIMRED_PCA):
    train_X_data, test_X_data = get_pca(train_X_data, test_X_data, pca_num)

if(DIMRED == DIMRED_LDA):
    train_X_data, test_X_data = get_lda(
        lda_num, train_X_data, test_X_data, train_Y_data)


# ------------
# Estimation

if(EST == EST_RF):
    print('Random Forst Training...')
    if GSCV:
        print('With GridSearchCV...')

        parameters = {'n_estimators': (10, 100, 1000)}
        est = RandomForestClassifier()
        est = grid_search.GridSearchCV(est, parameters)
        est = est.fit(train_X_data, train_Y_data)

        print("Best parameters set found on development set:")
        print()
        print(est.best_estimator_)
    else:
        est = RandomForestClassifier(n_estimators=100)
        est = est.fit(train_X_data, train_Y_data)

elif(EST == EST_LOGREG):
    print('Logistic Regression Training...')

    # "class_weight" parameter is specified to deal
    # with imbalanced data problem
    if GSCV:
        print('With GridSearchCV...')

        parameters = {'C': (0.001, 0.01, 0.1, 1)}
        est = linear_model.LogisticRegression(
            penalty='l1', class_weight='auto')
        est = grid_search.GridSearchCV(est, parameters)
        est = est.fit(train_X_data, train_Y_data)

        print("Best parameters set found on development set:")
        print()
        print(est.best_estimator_)
    else:
        est = linear_model.LogisticRegression(
            penalty='l1', class_weight='auto')
        est = est.fit(train_X_data, train_Y_data)

elif(EST == EST_SVC):
    print('SVC Training...')
    if GSCV:
        print('With GridSearchCV...')
        parameters = {'kernel': ('rbf', 'sigmoid'), 'C': [1, 10],
                      'gamma': [1.00000000e-06, 3.59381366e-06, 1.29154967e-05,
                                4.64158883e-05, 1.66810054e-04, 5.99484250e-04,
                                2.15443469e-03, 7.74263683e-03, 2.78255940e-02,
                                1.00000000e-01]}
        # parameters = {'kernel':('linear', 'rbf','poly','sigmoid')}
        # parameters = {'kernel':('rbf','sigmoid')}
        svc = svm.SVC()
        est = grid_search.GridSearchCV(svc, parameters)
        est.fit(train_X_data, train_Y_data)

        print("Best parameters set found on development set:")
        print()
        print(est.best_estimator_)
    else:
        knl = 'sigmoid'
        est = svm.SVC(kernel=knl)
        print('kernel is', knl)
    est = est.fit(train_X_data, train_Y_data)

elif(EST == EST_NN):
    print('NeuralNetwrok Training...')
    # Initialize Neural Network
    # #net = network.Network([len(train_X_data[0]), 50, 2])
    net = network2.Network(
        [len(train_X_data[0]), 10, 2], cost=network2.CrossEntropyCost())

    # Reshape Data for net.SGD
    temp_tr_X_data = []
    temp_tr_Y_data = []
    temp_ts_X_data = []
    # #temp_ts_srvs = []
    for i in range(0, len(train_X_data)):
        temp_tr_X_data.append(train_X_data[i][:, np.newaxis])
        if(train_Y_data[i] == 0):
            temp = np.array([1, 0])
        elif(train_Y_data[i] == 1):
            temp = np.array([0, 1])
        temp_tr_Y_data.append(temp[:, np.newaxis])
    for i in range(0, len(test_X_data)):
        temp_ts_X_data.append(test_X_data[i][:, np.newaxis])

    train_list = list(zip(temp_tr_X_data, temp_tr_Y_data))
    test_list = list(zip(temp_ts_X_data, test_Y_data))

    """
    //data_type example for SGD(from minst example)
    >>> tr_1 = list(tr_1)
    >>> te_1 = list(te_1)
    >>> tr_1[0][0].shape
    (784, 1)
    >>> tr_1[0][1].shape
    (10, 1)
    >>> te_1[0][1].shape
    ()
    >>> tr_1[0][1].shape
    (10, 1)
    >>>
    """
    # Let evaluation_data in net.SGD be test list when you plots graphs later
    # #net.SGD(train_list, 20, 10, 2, MULTI_FLAG, test_data=test_list)
    test_cost, test_accuracy, training_cost, training_accuracy = \
        net.SGD(train_list, 5, 20, 0.2, lmbda=1.0,
                evaluation_data=test_list,
                monitor_evaluation_accuracy=True,
                monitor_evaluation_cost=True,
                monitor_training_accuracy=True,
                monitor_training_cost=True)

    f = open("net_data.json", "w")
    json.dump([test_cost, test_accuracy, training_cost, training_accuracy], f)
    f.close()

    make_plots("net_data.json", 5,
               0, 0, 0, 0, len(test_Y_data), len(train_Y_data))

    print('Predicting...')
    prd_Y_data = net.prd_Y_data


if(EST != EST_NN):
    print('Predicting...')
    prd_Y_data = est.predict(test_X_data).astype(int)

predictions_file = open("EstimationResult.csv", "w", newline='')
open_file_object = csv.writer(predictions_file)
open_file_object.writerow(["PassengerId", "PredictedSurvived", "TrueSurvived"])
open_file_object.writerows(zip(ids, prd_Y_data, test_Y_data))
predictions_file.close()

# evaluates crossValidation Score
if(EST != EST_NN):
    get_Kfold_CrossValidation_score(train_X_data, train_Y_data, est, 3)

# evaluates prediction
get_evaluation(test_Y_data, prd_Y_data, [True, True, True, False, False])

# Plots SVC result
if(EST == EST_SVC and DIMRED == DIMRED_PCA and pca_num == 2):
    plot_svc_result(train_X_data, test_X_data, est)

print('Done...')

"""
LOGREGL2:
array([[-0.59046734, -0.0225814 , -0.30679091,  0.05951638,  0.00259083,
        -0.04371637, -2.46999059]])
LOGREGL1:
[-0.800, -0.03, -0.31, 0.02, 0.0,
-0.08, -2.66]
"""
