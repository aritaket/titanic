""" Writing my first randomforest code.
Author : Yuzuru Kato
Date : 2th Feb 2015
"""

# import third party libraries

# for treat numerical data
import pandas as pd
import numpy as np

# for oversampling and undersampling
import random
from random import choice
from sklearn.neighbors import NearestNeighbors

# --------------------
# for pre processing

# get ports is requrired since we need to get all unique values
# in advance


def get_ports(train_reader, test_reader, keys, y_key):
    """ get Ports(array containing unique values for each variable)
    param train_reader: train_reader with chank size specifier
    return Ports: array contaning unique values for each variable
    """
    # get empty list for each variable
    ports = []
    for i in range(0, len(keys)):
        if(keys[i] != y_key):  # y_key is not needed
            key_port = []
            ports.append(key_port)
        else:
            del_index = i

    del keys[del_index]

    # get unique values from each chank
    count = 0

    # from training data
    print("train reading...")
    for read_df in train_reader:
        for i in range(0, len(keys)):
            # get new unique values
            ports[i].extend(list(np.unique(read_df[keys[i]])))
            # store only unique values(delete overlapped data)
            ports[i] = list(np.unique(ports[i]))

        print(count)
        count += 1

    count = 0

    # from test data
    print("test reading...")
    for read_df in test_reader:
        for i in range(0, len(keys)):
            # get new unique values
            ports[i].extend(list(np.unique(read_df[keys[i]])))
            # store only unique values(delete overlapped data)
            ports[i] = list(np.unique(ports[i]))

        print(count)
        count += 1

    # Converts each port to numpy for saving as j-son
    for i in range(0, len(keys)):
        ports[i] = np.array(ports[i])

    return ports


def converts_to_num(train_df, ports, keys, converted_variables):
    """converts objects into numerical values
    param train_df: training data_frame
    param prots: unique variables for all variables
    param keys: names list for training data_frame
    param converted_variables: converted variables list
    return: converted train_df
    """
    for i in range(0, len(keys)):

        # converts only variables in conveted varialbes list
        CONV_FLAG = False
        for j in range(0, len((converted_variables))):
            if keys[i] == converted_variables[j]:
                CONV_FLAG = True

        if CONV_FLAG:
            key = keys[i]
            port = list(enumerate(np.unique(ports[i])))

            # in case when you want to convert
            # only non int or float data
            """
            if(train_df[key].dtypes == 'int64'
               or train_df[key].dtypes == 'int32'
               or train_df[key].dtypes == 'float64'
               or train_df[key].dtypes == 'float32'
              ):
                TY_NUM = True
            else:
                TY_NUM = False
            if not TY_NUM:
            """
            ports_dict = {name: i for i, name in port}

            # Converts each value in each variable i
            # into indexs in the variable.
            train_df[key] = train_df[key].map(
                lambda x: ports_dict[x]).astype(int)

    return train_df


def values_to_binary(train_df, ports, keys, converted_variables,
                     binary_variables):
    """convert values in variables(train_df[key]) into
    a binary matrix.
    param train_df: training data_frame
    param prots: unique variables for all variables
    param keys: names list for training data_frame
    param converted_variables: converted variables list
    param binaly_variables: variables list conveted to binary
    return: converted train_df
    """
    for i in range(0, len(keys)):

        # converts only variables in binary varialbes list
        BINARY_FLAG = False
        for j in range(0, len((binary_variables))):
            if keys[i] == binary_variables[j]:
                BINARY_FLAG = True

        # converts only variables in conveted varialbes list
        CONV_FLAG = False
        for j in range(0, len((converted_variables))):
            if keys[i] == converted_variables[j]:
                CONV_FLAG = True

        if BINARY_FLAG:
            # gets zeros matrix
            key = keys[i]
            port = list(enumerate(np.unique(ports[i])))

            key_np = np.zeros(
                len(train_df[key]) * len(port)).reshape(len(train_df[key]),
                                                        len(port))

            # INDEX 0: conveted num(index for each unique value)
            # INDEX 1: true value (only num)
            if CONV_FLAG:
                INDEX = 0
            else:
                INDEX = 1

            # insert 1 where each value exists
            train_np = train_df[key].values
            for j in range(0, len(port)):
                key_np[train_np == port[j][INDEX], j] = 1

            # covert numpy array to data frame (+ add name)
            key_df_columns = list()
            for j in range(0, len(port)):
                # give name to each binary variables [j][1] means that
                # we use real values for names
                key_df_columns.append("{0}:{1}".format(key, port[j][1]))
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
