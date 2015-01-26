""" click_estimator code.
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

# ------------
# Define Parameters

# Sets parameters
TR_NUM = 5000       # Number Of Train data Size to be red
# TE_NUM = 1000;    # Number Of Test data Size to be red
# SUB_NUM = 1000;   # Number Of Submission data to be red
CUT_TEST_NUM = TR_NUM / 10  # Number of test data size

# Define Estimatior
EST_RF = 0
EST_SVC = 1
EST_LOGREG = 2
EST_NN = 3
EST = EST_NN

# For RF,SVC,LOGREG:Whether or not GridSearchCV is performed (currently
# only for SVC)
GSCV = False

# Whether or not PCA or LDA is performed
DIMRED_NONE = 0
DIMRED_PCA = 1
DIMRED_LDA = 2
DIMRED = DIMRED_NONE

pca_num = 5
lda_num = 5

# Wheather daata is impblanced or not(performs SMOTE or not)
IMBALANCE = False

# ------------
# Reads DATA

# When you use "chunksize" specifier, pandas reads data as TextReader
train_reader = pd.read_csv("train.csv", header=0, chunksize=TR_NUM)
# #test_reader = pd.read_csv("test.csv", header=0, chunksize=TE_NUM)
# #sub_reader = pd.read_csv("sampleSubmission.csv", header=0,
# #chunksize=SUB_NUM)

# Transform TextReader into Data Frame
train_df = train_reader.get_chunk(TR_NUM)
# #test_df = test_reader.get_chunk(TE_NUM);
# #sub_df = sub_reader.get_chunk(SUB_NUM);

# Output data to visualize the ratio of ON and OFF for each unique value
# in each variable
VISUALIZE = False
PLOT = False
if VISUALIZE:
    output_data(train_df, PLOT)

# ------------
# Pre-Processing

# Transform data frame into numpy array
# Collects train_Y_data, test(true)_Y_data,ids
train_Y_data = train_df['click'].values[:-CUT_TEST_NUM]
test_Y_data = train_df['click'].values[-CUT_TEST_NUM:]
ids = train_df['id'].values[-CUT_TEST_NUM:]

# Remove "id" and "click" which are not used in training data
train_df = train_df.drop(['id', 'click'], axis=1)

# Drop unused varialbes
used_variables_list = ['C16', 'C15', 'C18', 'C20', 'C14', 'C17', 'C19']
# #used_variables_list = ['site_category','app_category', 'device_type']
# #used_variables_list = ['C16', 'C18']
train_df = drop_unused_variables(train_df, used_variables_list)

# Converts object into numerical values
converts_obj_to_num(train_df)

# Chnge values into binary data
for j in range(0, len(used_variables_list)):
    train_df = values_to_binary(train_df, used_variables_list[j])

# Get TRAIN DATA and Test DATA
train_X_data = train_df.values[:-CUT_TEST_NUM, :]
test_X_data = train_df.values[-CUT_TEST_NUM:, :]

# Performs over_sampling and under_sampling
if IMBALANCE:
    O_N = 200
    O_k = 4
    U_N = 80
    train_X_data, train_Y_data = get_smote(
        train_X_data, train_Y_data, O_N, O_k, U_N)

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

# ------------
# Output and Evaluation

predictions_file = open("EstimationResult.csv", "w", newline='')
open_file_object = csv.writer(predictions_file)
open_file_object.writerow(["id", "Prd_click", "Tr_click"])
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