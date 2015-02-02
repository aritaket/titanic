# third party library
import pandas as pd
import numpy as np
import csv as csv
# import matplotlib.pyplot as plt
import json
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB

# My library
from tools_for_submit import *

# from sklearn.decomposition import IncrementalPCA
# -----------------------------
# Define Parameters

# Sets parameters
CHUNK_NUM = 1000       # Number Of Train data Size to be red

# Define Estimatior
EST_NB_MT = 0
EST_NB_BE = 1

EST = EST_NB_MT

"""
DIMRED_NONE = 0
DIMRED_PCA = 1

DIMRED == DIMRED_PCA
"""

# Wheather data is impblanced or not(performs SMOTE or not)
IMBALANCE = True

# -----------------------------
# Gets key and port data (performed once and
# saved as json format)
# Gets keys
train_reader = pd.read_csv("train_part.csv", header=0, chunksize=1)
train_df = train_reader.get_chunk(1)
keys = list(train_df)

# get ports
train_reader = pd.read_csv("train_part.csv", header=0, chunksize=CHUNK_NUM)
test_reader = pd.read_csv("test_part.csv", header=0, chunksize=CHUNK_NUM)
y_key = "click"
ports = get_ports(train_reader, test_reader, keys, y_key)

# save as json
data = {"keys": keys,
        "ports": [port.tolist() for port in ports],
        }

f = open("key_port.json", "w")
json.dump(data, f)
f.close()
# -----------------------------
# load json
f = open("key_port.json", "r")
data = json.load(f)
f.close()

keys = data["keys"]
ports = [np.array(port) for port in data["ports"]]

# ------------------------------
# define parameters
binary_variables = ['C16', 'C15', 'C18',
                    'C20', 'C17', 'C19', 'C1', 'C21',
                    'site_domain', 'app_domain',
                    'device_type', 'device_conn_type',
                    'banner_pos']

# converted_variables = [] (currently included in the loop)

# currently ipca is not implemented
"""
if(DIMRED == DIMRED_PCA):
    ipca = IncrementalPCA(n_components=pca_num)
"""

# defines model
if(EST == EST_NB_MT):
    est = MultinomialNB()
if(EST == EST_NB_BE):
    est = BernoulliNB()

# -------------------------------------
print("training...")
# get train_reader
train_reader = pd.read_csv("train_part.csv", header=0, chunksize=CHUNK_NUM)
count = 0
for read_df in train_reader:

    # ------------
    # Pre-Processing
    train_df = read_df

    # Gets train_Y_data
    train_Y_data = train_df['click'].values

    # Removes "id" and "click" which are not used in training data
    train_df = train_df.drop(['id', 'click'], axis=1)

    # get converted_variables
    converted_variables = list(train_df)

    # convert values into dictionary number
    train_df = converts_to_num(train_df, ports, keys, converted_variables)

    # Chnge values into binary data
    train_df = values_to_binary(train_df, ports, keys, converted_variables,
                                binary_variables)

    # Get train X_data
    train_X_data = train_df.values

    # Performs over_sampling and under_sampling
    if IMBALANCE:
        O_N = 300
        O_k = 4
        U_N = 70
        train_X_data, train_Y_data = get_smote(
            train_X_data, train_Y_data, O_N, O_k, U_N)

    """ dimension reduciton
    # only checked that the programming is working
    if(DIMRED == DIMRED_PCA):
        ipca.partial_fit(train_X_data)
        train_X_data = ipca.transform(train_X_data)
        print("PCA_performed")
    """

    # ------------
    # Estimation (partial_fit is used when you want to make
    # the model learn incrementaly)
    if(EST == EST_NB_MT):
        print(' NB_MT Training...')
        est = est.partial_fit(train_X_data, train_Y_data, classes=[0, 1])
    if(EST == EST_NB_BE):
        print(' NB_BE Training...')
        est = est.partial_fit(train_X_data, train_Y_data, classes=[0, 1])

    print(count)
    count += 1

# ------------
# Output test data


# Reads test data
test_reader = pd.read_csv("test_part.csv", header=0, chunksize=CHUNK_NUM)

count = 0  # for checking progress
predictions_file = open("Submit.csv", "w", newline='')
open_file_object = csv.writer(predictions_file)
open_file_object.writerow(["id", "click"])

for read_df in test_reader:
    test_df = read_df
    ids = test_df['id'].values

    # -----------------
    # Pre-processing
    test_df = test_df.drop(['id'], axis=1)

    converted_variables = list(test_df)

    # Converts object into numerical values
    test_df = converts_to_num(test_df, ports, keys, converted_variables)

    # Chnge values into binary data
    test_df = values_to_binary(test_df, ports, keys, converted_variables,
                               binary_variables)

    # Get Test DATA
    test_X_data = test_df.values

    """
    if(DIMRED == DIMRED_PCA):
        print("PCA_performed")
        test_X_data = ipca.transform(test_X_data)
    """

    # ----------------
    # Predicting data
    if(EST == EST_NB_MT):
        print('NB_MT Predicting...')
    elif(EST == EST_NB_BE):
        print('NB_BE Predicting...')
    test_Y_data = est.predict(test_X_data).astype(int)

    # Output result for test data
    open_file_object.writerows(zip(ids, test_Y_data))

    print(count)
    count += 1

predictions_file.close()
print("Done")
