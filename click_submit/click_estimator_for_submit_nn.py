# third party library
import pandas as pd
import numpy as np
import csv as csv
# import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
import json

# my library
from tools_for_submit import *

# Neilsen's library
from network2_for_submit import *
from network_plot import *

# from sklearn.decomposition import IncrementalPCA
# -----------------------------
# Define Parameters

# Sets parameters
CHUNK_NUM = 5000       # Number Of Train data Size to be red

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

# define binary_variables
binary_variables = ['C16', 'C15', 'C18',
                    'C20', 'C17', 'C19', 'C1', 'C21',
                    'site_domain', 'app_domain',
                    'device_type', 'device_conn_type',
                    'banner_pos']

# converted_variables = [] (currently included in the loop)

print("training...")
epochs = 10  # epoch for network training

for epoch in range(0, epochs):
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
        # Get TRAIN DATA
        train_X_data = train_df.values

        # Performs over_sampling and under_sampling
        if IMBALANCE:
            O_N = 300
            O_k = 4
            U_N = 70
            train_X_data, train_Y_data = get_smote(
                train_X_data, train_Y_data, O_N, O_k, U_N)

        # ---------------------------
        # Estimation

        if(count == 0 and epoch == 0):
            # defines network only at the first step
            net = Network(
                [len(train_X_data[0]), 30, 2], cost=CrossEntropyCost())
        else:
            # loads network in previous step
            net = load("net.json", net)

        # Reshape Data for net.SGD
        temp_tr_X_data = []
        temp_tr_Y_data = []
        # #temp_ts_srvs = []
        for i in range(0, len(train_X_data)):
            temp_tr_X_data.append(train_X_data[i][:, np.newaxis])
            if(train_Y_data[i] == 0):
                temp = np.array([1, 0])
            elif(train_Y_data[i] == 1):
                temp = np.array([0, 1])
            temp_tr_Y_data.append(temp[:, np.newaxis])

        train_list = list(zip(temp_tr_X_data, temp_tr_Y_data))

        test_cost, test_accuracy, training_cost, training_accuracy = \
            net.SGD(train_list, epoch, 20, 0.001, lmbda=3.0,
                    evaluation_data=None,
                    monitor_evaluation_accuracy=False,
                    monitor_evaluation_cost=False,
                    monitor_training_accuracy=True,
                    monitor_training_cost=True,
                    )

        # save current network data
        net.save("net.json")

        print(count)
        print("--------")
        count += 1
    print("complete epoch")
    print(epoch)

# ------------
# Output and Evaluation

# READ test data
test_reader = pd.read_csv("test_part.csv", header=0, chunksize=CHUNK_NUM)

count = 0  # for check progress
predictions_file = open("Submit.csv", "w", newline='')
open_file_object = csv.writer(predictions_file)
open_file_object.writerow(["id", "click"])

for read_df in test_reader:
    test_df = read_df
    ids = test_df['id'].values

    # -----------------
    # Predicting data
    test_df = test_df.drop(['id'], axis=1)

    converted_variables = list(test_df)

    # Converts object into numerical values
    test_df = converts_to_num(test_df, ports, keys, converted_variables)

    # Chnge values into binary data
    test_df = values_to_binary(test_df, ports, keys, converted_variables,
                               binary_variables)

    # Get Test DATA
    test_X_data = test_df.values

    # Reshape Data for net.SGD
    temp_ts_X_data = []
    for i in range(0, len(test_X_data)):
        temp_ts_X_data.append(test_X_data[i][:, np.newaxis])
    test_list = temp_ts_X_data

    print("net predicting")
    net.predict(test_list)
    test_Y_data = net.prd_Y_data

    # Output result for test data
    open_file_object.writerows(zip(ids, test_Y_data))

    print(count)
    count += 1

predictions_file.close()
print("Done")
