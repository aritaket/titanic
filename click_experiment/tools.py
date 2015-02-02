""" Writing my first randomforest code.
Author : Yuzuru Kato
Date : 18th Jan 2015
"""

# import third party libraries

# for treat numerical data
import pandas as pd
import numpy as np
import csv as csv

# for evaluation
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import auc
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt

# for pca and lda
from sklearn import decomposition
from sklearn.lda import LDA

# for oversampling and undersampling
import random
from random import choice
from sklearn.neighbors import NearestNeighbors

# --------------------
# for pre processing


def converts_to_num(train_df):
    """converts objects into numerical values
    param train_df: training_dataframe
    return: pre_processed train_df
    """
    for i in range(0, len(list(train_df))):
        key = list(train_df)[i]
        # in case when integer and float data are not covnverted
# #        if(train_df[key].dtypes == 'int64'
# #           or train_df[key].dtypes == 'int32'
# #           or train_df[key].dtypes == 'float64'
# #           or train_df[key].dtypes == 'float32'
# #           ):
# #            TY_NUM = True
# #        else:
# #            TY_NUM = False
# #        # print("{0}:{1}".format(Key,TY_INT))
# #        # Converts string to int
# #        if not TY_NUM:
        Ports = list(enumerate(np.unique(train_df[key])))
        Ports_dict = {name: i for i, name in Ports}

        # Converts each value in each variable into indexs in the variable.
        train_df[key] = train_df[key].map(
            lambda x: Ports_dict[x]).astype(int)
    return train_df


def values_to_binary(train_df, key):
    """convert values in variable(train_df[key]) into a binary matrix.
    param train_df: training data frame
    param key: key for converting variable
    return: converted train_df
    """
    # converts variables into binary data
    # gets all values
    Ports = list(enumerate(np.unique(train_df[key])))

    # gets zeros matrix
    key_np = np.zeros(
        len(train_df[key]) * len(Ports)).reshape(len(train_df[key]),
                                                 len(Ports))
    # insert 1 where each value exists
    train_np = train_df[key].values
    for j in range(0, len(Ports)):
        # numbers in Prots[:,1] are real values
        key_np[train_np == Ports[j][1], j] = 1

    # covert numpy array to data frame (+ add name)
    key_df_columns = list()
    for j in range(0, len(Ports)):
        key_df_columns.append("{0}:{1}".format(key, Ports[j][1]))
    key_df = pd.DataFrame(key_np, columns=key_df_columns)

    # concatenate key_df to train_df
    train_df = pd.concat([train_df, key_df], axis=1)

    # delete key value
    train_df.drop(key, axis=1, inplace=True)

    return train_df


def drop_unused_variables(train_df, used_variables_list):
    """Drop unused variable
    param train_df: train dataframe
    param used_variable_list: list of variables which we use
    retrun: train_df whose unused variables are dropped
    """
    length = len(list(train_df))
    dr_idx = 0
    for i in range(0, length):
        DROP = True
        for j in range(0, len(used_variables_list)):
            if(list(train_df)[dr_idx] == used_variables_list[j]):
                DROP = False

        if not DROP:
            # #print("{0} is undrop".format(list(train_df)[dr_idx]))
            dr_idx += 1  # UnDrop used variables
        elif DROP:
            # #print("{0} is drop".format(list(train_df)[dr_idx]))
            # Drop unused variables
            train_df.drop(list(train_df)[dr_idx], axis=1, inplace=True)
    return train_df


def over_sampling(T, N, k):
    """
    we uses SMOTE for over_sampling. This code is refered from
    http://comments.gmane.org/gmane.comp.python.scikit-learn/5278
    prama T:array-like, shape = [n_minority_samples, n_features]
    Holds the minority samples
    param N:percetange of new synthetic samples:
    n_synthetic_samples = N/100 * n_minority_samples.
    param k:number of nearest neighbours
    reeturn S:(N/100) * n_minority_samples synthetic minority samples.
    """
    n_minority_samples, n_features = T.shape

    if (N % 100) != 0:
        raise ValueError("N must be < 100 or multiple of 100")

    N = int(N / 100)
    n_synthetic_samples = N * n_minority_samples
    S = np.zeros(shape=(n_synthetic_samples, n_features))

    # Learn nearest neighbours
    neigh = NearestNeighbors(n_neighbors=k)
    neigh.fit(T)

    # Calculate synthetic samples
    for i in range(n_minority_samples):
        nn = neigh.kneighbors(T[i], return_distance=False)
        for n in range(N):
            nn_index = choice(nn[0])
            # NOTE: nn includes T[i], we don't want to select it
            while nn_index == i:
                nn_index = choice(nn[0])

            dif = T[nn_index] - T[i]
            gap = np.random.random()
            S[n + i * N, :] = T[i, :] + gap * dif[:]
    return S


def under_sampling(T, N):
    """perform under_sampling_data to impbalanced data
    prama T:array-like, shape = [n_majority_samples, n_features]
    Holds the majority samples
    param N:percetange of samples to get:
    samples = N/100 * n_majority_samples.
    return T:(N/100) * n_majority_samples.
    """
    n_majority_samples, n_features = T.shape

    zero_one_mat = np.zeros(n_majority_samples)
    i = 0
    while i < int(N / 100 * n_majority_samples):
        # length - i is current T length
        index = np.random.choice(n_majority_samples, 1)
        if(zero_one_mat[index] == 0):
            zero_one_mat[index] = 1
            i += 1
        # if(i % 1000 == 0):
        #    print(i)
    # print(zero_one_mat)
    T = T[zero_one_mat == 1, :]
    return T


def get_smote(train_X_data, train_Y_data, O_N, O_k, U_N):
    """perform over and under_sampling_data to impbalanced data
    prama train_X_data:train_X_data
    prama train_Y_data:train_X_data
    param O_N:  N for over_sampling
    praam O_k:  k for over_sampling
    param U_N:  N for over_undersampling
    return train_X_data: smoted train_X_data
    train_Y_data: smoted train_Y_data
    """
    L_0 = len(train_X_data[train_Y_data == 0])
    L_1 = len(train_X_data[train_Y_data == 1])

    if(L_0 < L_1):
        minority_data = train_X_data[train_Y_data == 0]
        majority_data = train_X_data[train_Y_data == 1]
    elif(L_1 <= L_0):
        minority_data = train_X_data[train_Y_data == 1]
        majority_data = train_X_data[train_Y_data == 0]

    minority_data = over_sampling(minority_data, O_N, O_k)

    majority_data = under_sampling(majority_data, U_N)
    ##
    train_X_data = np.r_[minority_data, majority_data]
    # Update train_Y_data also. Some error occurs
    if(L_0 < L_1):
        train_Y_data = np.r_[
            np.zeros(len(minority_data)), np.ones(len(majority_data))]
    elif(L_1 <= L_0):
        train_Y_data = np.r_[
            np.ones(len(minority_data)), np.zeros(len(majority_data))]
    temp_data = list(np.c_[train_X_data, train_Y_data])
    # random shuffle only works for list
    # datahttp://d.hatena.ne.jp/haru_ton/20100401/1270124407
    random.shuffle(temp_data)
    temp_data = np.array(temp_data)
    train_X_data = temp_data[:, :-1]
    # Change shape as (len,) for estimator's argument.
    train_Y_data = temp_data[:, -1:].reshape(len(temp_data),)
    return train_X_data, train_Y_data


def get_pca(train_X_data, test_X_data, pca_num):
    """Performes pca with pca_num's components to X_data.
    param train_X_data: train_X_data to which PCA is performed
    param test_X_data: PCA should also be perforrmed to test_X_data
    param pca_num: number of components to keep
    retrun: train_X_data and test_X_data to which PCA is performed
    """
    pca = decomposition.PCA()
    pca.n_components = pca_num
    pca.fit(train_X_data)  # .transform(X_data)
    train_X_data = pca.transform(train_X_data)
    test_X_data = pca.transform(test_X_data)
    print("PCA_performed")
    print(pca.explained_variance_)
    # print(pca.components_)
    return train_X_data, test_X_data


def get_lda(lda_num, train_X_data, test_X_data, train_Y_data=None):
    """
    Performs lda with lda_num's components to X_data.
    param lda_num: number of components to keep
    param X_data: X_data to which LDA is performed
    param Y_data: Y_data to which LDA is performed
    retrun: train_X_data and test_X_data to which LDA is performed
    """
    lda = LDA()
    lda.n_components = lda_num
    lda.fit(train_X_data, train_Y_data)
    train_X_data = lda.transform(train_X_data)
    test_X_data = lda.transform(test_X_data)
    print("LDA_is performed")
    return train_X_data, test_X_data

# --------------------
# for evaluation


def get_evaluation(test_Y_data, prd_Y_data, TF_list):
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
    # Prints Confusion Matrix
    if TF_list[0]:
        print(confusion_matrix(test_Y_data, prd_Y_data))

    # Prints Accuracy
    if TF_list[1]:
        print(accuracy_score(test_Y_data, prd_Y_data))

    # Prints P,R,F,S values
    if TF_list[2]:
        target_names = ['0', '1']
        print(classification_report(
            test_Y_data, prd_Y_data, target_names=target_names))

    # plots precision-recall curve
    if TF_list[3]:
        precision, recall, pr_thresholds = precision_recall_curve(
            test_Y_data, prd_Y_data)
        area = auc(recall, precision)
        print("Area Under Curve: %0.2f" % area)
        plt.figure(0)
        plt.clf()
        plt.plot(recall, precision, label='Precision-Recall curve')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.ylim([0.0, 1.05])
        plt.xlim([0.0, 1.0])
        plt.title('Precision-Recall example: AUC=%0.2f' % area)
        plt.legend(loc="lower left")

    # Plots ROC curve
    if TF_list[4]:
        fpr, tpr, roc_thresholds = roc_curve(test_Y_data, prd_Y_data)
        roc_auc = auc(fpr, tpr)
        print("Area under the ROC curve : %f" % roc_auc)
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

    if TF_list[3] or TF_list[4]:
        plt.show()


def get_Kfold_CrossValidation_score(train_X_data, train_Y_data, est, fold_num):
    """ Outputs CrossValidationScore
    param train_X_data: train_X_data
    param train_Y_data: train_Y_data
    param est: Estimatior
    param fold: the Value of K
    """
    # Prints cross_validation score
    X_folds = np.array_split(train_X_data, fold_num)
    y_folds = np.array_split(train_Y_data, fold_num)
    scores = list()
    for k in range(fold_num):
        # We use 'list' to copy, in order to 'pop' later on
        X_train = list(X_folds)
        X_test = X_train.pop(k)
        X_train = np.concatenate(X_train)
        y_train = list(y_folds)
        y_test = y_train.pop(k)
        y_train = np.concatenate(y_train)
        scores.append(est.fit(X_train, y_train).score(X_test, y_test))
    print(scores)


def plot_svc_result(train_X_data, test_X_data, svc):
    """ Outputs SVC results with 2 PCA components
    param X_train: X_train_data
    param X_test: X_test_data
    param svc: Svc
    """
    plt.figure(0)
    plt.clf()
    plt.scatter(
        train_X_data[:, 0], train_X_data[:, 1], zorder=10, cmap=plt.cm.Paired)

    # Circle out the test data
    plt.scatter(test_X_data[:, 0], test_X_data[:, 1],
                s=80, facecolors='none', zorder=10)

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
    plt.contour(XX, YY, Z, colors=['k', 'k', 'k'],
                linestyles=['--', '-', '--'],
                levels=[-.5, 0, .5])

    plt.title('SCV_result')
    plt.show()

# --------------------
# for output data structure


def output_data(train_df, PLOT):
    """ Outputs total, total_click_on,total_click_off
    for each values for each variables as csv
    param train_df: train_dataframe
    param PLOT: FLAG for whether or not we plot data.
    """
    train_df_click_on = train_df[train_df['click'] == 1]
    train_df_click_off = train_df[train_df['click'] == 0]

    train_df = train_df.drop(['id', 'click'], axis=1)
    train_df_click_on = train_df_click_on.drop(['id', 'click'], axis=1)
    train_df_click_off = train_df_click_off.drop(['id', 'click'], axis=1)

    for i in range(0, len(list(train_df))):
        key = list(train_df)[i]
        if(train_df[key].dtypes == 'int64'):
            TY_INT = True
        else:
            TY_INT = False

        # Converts string to int
        Ports = list(enumerate(np.unique(train_df[key])))
        if not TY_INT:
            Ports_dict = {name: i for i, name in Ports}
            train_df[key] = train_df[key].map(
                lambda x: Ports_dict[x]).astype(int)
            train_df_click_on[key] = train_df_click_on[
                key].map(lambda x: Ports_dict[x]).astype(int)
            train_df_click_off[key] = train_df_click_off[
                key].map(lambda x: Ports_dict[x]).astype(int)

        # Get Total Num for each value in each variable
        num_port = np.zeros(len(Ports))
        num_port_on = np.zeros(len(Ports))
        num_port_off = np.zeros(len(Ports))
        ratio_on = np.zeros(len(Ports))
        ratio_off = np.zeros(len(Ports))

        # Ports(key,RealValues)
        for j in range(0, len(Ports)):
            if TY_INT:
                index = 1  # Gets Real int Values
            elif not TY_INT:
                index = 0  # Gets Index for String Values
            num_port[j] = len(train_df[key][train_df[key] ==
                                            Ports[j][index]])
            num_port_on[j] = len(
                train_df_click_on[key][train_df_click_on[key] ==
                                       Ports[j][index]])
            num_port_off[j] = len(
                train_df_click_off[key][train_df_click_off[key] ==
                                        Ports[j][index]])
            ratio_on[j] = num_port_on[j] / num_port[j]
            ratio_off[j] = num_port_off[j] / num_port[j]

        # Plot Total (Zero,One) Num and One Num
        if PLOT:
            plt.figure(i)
            plt.clf()

            x = np.array(Ports)[:, index]
            y1 = num_port
            y2 = num_port_on
            plt.title("Scatters of {0}".format(key))
            plt.plot(x, y1, "ro")
            plt.plot(x, y2, "bo")
            plt.show()

        # Outputs Data as Files
        # print(key)
        # print(len(Ports))
        # print(len(num_port))
        # print(len(num_port_on))
        # print(len(num_port_off))
        print("Finish {0}".format(key))
        predictions_file = open("./Files/{}.csv".format(key), "w",
                                newline='')
        open_file_object = csv.writer(predictions_file)
        open_file_object.writerow(
            ["key", "real_value", "T_Num", "On_Num", "Off_Num",
             "ON_Ratio", "Off_Ratio"])
        open_file_object.writerows(zip(np.array(Ports)[:, 0], np.array(
            Ports)[:, 1], num_port, num_port_on, num_port_off,
            ratio_on, ratio_off))
        open_file_object.writerow(["-", "SUM", sum(num_port), sum(num_port_on),
                                   sum(num_port_off), sum(num_port_on) /
                                   sum(num_port), sum(num_port_off) /
                                   sum(num_port)])
        predictions_file.close()
