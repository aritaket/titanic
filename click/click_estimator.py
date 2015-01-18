""" Writing my first randomforest code.
Author : Yuzuru Kato
Date : 12rd Jan 2015
""" 
import pandas as pd
import numpy as np
import csv as csv

import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn import linear_model
from sklearn import grid_search

from tools import *

def output_data(train_df, PLOT):

    train_df_click_on =   train_df[train_df['click'] == 1]
    train_df_click_off =  train_df[train_df['click'] == 0]

    train_df = train_df.drop(['id','click'], axis=1) 
    train_df_click_on = train_df_click_on.drop(['id','click'], axis=1) 
    train_df_click_off = train_df_click_off.drop(['id','click'], axis=1) 
    
    for i in range(0,len(list(train_df))):
        key =list(train_df)[i]
        if(train_df[key].dtypes == 'int64'):
            TY_INT = True
        else:
            TY_INT = False

        #Converts string to int
        Ports = list(enumerate(np.unique(train_df[key])))
        if(TY_INT == False):
            Ports_dict = { name : i for i, name in Ports }
            train_df[key] = train_df[key].map( lambda x: Ports_dict[x]).astype(int)
            train_df_click_on[key] = train_df_click_on[key].map( lambda x: Ports_dict[x]).astype(int)
            train_df_click_off[key] = train_df_click_off[key].map( lambda x: Ports_dict[x]).astype(int)

        #Get Total Num for each value in each variable
        num_port = np.zeros(len(Ports))
        num_port_on = np.zeros(len(Ports))
        num_port_off = np.zeros(len(Ports))
        ratio_on = np.zeros(len(Ports))
        ratio_off = np.zeros(len(Ports))

        # Ports(key,RealValues)
        for j in range(0,len(Ports)):
            if(TY_INT == True):
                index = 1;  ## Gets Real int Values
            elif(TY_INT == False):
                index = 0;  ## Gets Index for String Values           
            num_port[j] = len(train_df[key][train_df[key] == Ports[j][index]])    
            num_port_on[j] = len(train_df_click_on[key][train_df_click_on[key] == Ports[j][index]])
            num_port_off[j] = len(train_df_click_off[key][train_df_click_off[key] == Ports[j][index]])
            ratio_on[j] = num_port_on[j]/num_port[j]
            ratio_off[j] = num_port_off[j]/num_port[j] 
            
        ##Plot Total (Zero,One) Num and One Num
        if(PLOT == True):
            plt.figure(i) 
            plt.clf()

            x = np.array(Ports)[:,index]
            y1 = num_port
            y2 = num_port_on
            plt.title("Scatters of {0}".format(key))
            plt.plot(x, y1, "ro")
            plt.plot(x, y2, "bo")
            plt.show()
            
        #Outputs Data as Files
##        print(key)
##        print(len(Ports))
##        print(len(num_port))
##        print(len(num_port_on))
##        print(len(num_port_off))
        print("Finish {0}".format(key))
        predictions_file = open("./Files/{}.csv".format(key), "w",newline='')
        open_file_object = csv.writer(predictions_file)
        open_file_object.writerow(["key","real_value","T_Num","On_Num","Off_Num","ON_Ratio","Off_Ratio" ])
        open_file_object.writerows(zip( np.array(Ports)[:,0], np.array(Ports)[:,1],num_port,num_port_on,num_port_off,ratio_on,ratio_off))
        open_file_object.writerow(["-","SUM",sum(num_port),sum(num_port_on),sum(num_port_off),sum(num_port_on)/sum(num_port),sum(num_port_off)/sum(num_port)])
        predictions_file.close()




 
#Sets parameters
TR_NUM = 100000;     #Number Of Train data Size to be red
#TE_NUM = 1000;     #Number Of Test data Sizeto to be red
#SUB_NUM = 1000;    #Number Of Submission data to be red

CUT_TEST_NUM = TR_NUM/10;

#NAMEs for Estimator
EST_RF = 0
EST_SVC = 1
EST_LOGREG =2
EST_NN =3

#Define Estimatior
EST = EST_LOGREG  

#Whether or not PCA is performed
PCA = False      
pca_num = 2

#Whether or not GridSearchCV is performed (currently only for SVC)
GSCV = False

# Reads DATA
# When you use "chunksize" specifier, pandas reads data as TextReader
train_reader = pd.read_csv("train.csv", header=0, chunksize=TR_NUM)
##test_reader = pd.read_csv("test.csv", header=0, chunksize=TE_NUM)
##sub_reader = pd.read_csv("sampleSubmission.csv", header=0, chunksize=SUB_NUM)

# Transform TextReader into Data Frame
train_df = train_reader.get_chunk(TR_NUM);
##test_df = test_reader.get_chunk(TE_NUM);
##sub_df = sub_reader.get_chunk(SUB_NUM);

# Output data to visualize the ratio of ON and OFF for each unique value in each variable 
VISUALIZE = False;
PLOT = False;
if(VISUALIZE == True):
    output_data(train_df,PLOT)

#Transform data frame into numpy array
#Collects train_Y_data, test(true)_Y_data,ids
train_Y_data = train_df['click'].values[:-CUT_TEST_NUM]
test_Y_data = train_df['click'].values[-CUT_TEST_NUM:]
ids = train_df['id'].values[-CUT_TEST_NUM:]

#Remove "id" and "click" which are not used in training data
train_df = train_df.drop(['id','click'], axis=1) 
    
#Pre-processes the data frame
pre_processing(train_df)

#Drop unused varialbes
used_variables_list = ['C16','C15', 'C18', 'C20' ,'C14', 'C17', 'C19']
train_df = drop_unused_variables(train_df, used_variables_list)

#Get TRAIN DATA
train_X_data = train_df.values[:-CUT_TEST_NUM,:]
if(PCA == True):
    train_X_data, pca_components = get_pca(train_X_data,pca_num)

#Get TEST DATA
test_data = train_df.values[-CUT_TEST_NUM:,:]
if(PCA == True):
    test_X_data = test_X_data.dot(pca_components.T)

CALC = True;
if(CALC == True):         
    if(EST == EST_RF):
        print ('Random Forst Training...')
        if(GSCV == True):
            print ('With GridSearchCV...')

            parameters ={'n_estimators':(10,100,1000)}
            est = RandomForestClassifier()
            est = grid_search.GridSearchCV(est, parameters)
            est = est.fit(train_X_data,train_Y_data)

            print("Best parameters set found on development set:")
            print()
            print(est.best_estimator_)
        else:        
            est = RandomForestClassifier(n_estimators=100)
            est = est.fit(train_X_data,train_Y_data)

    elif(EST == EST_LOGREG):
        print('Logistic Regression Training...')
        ## "class_weight" parameter is specified to deal with imbalanced data problem

        if(GSCV == True):
            print ('With GridSearchCV...')
            
            parameters = {'C':(0.001,0.01,0.1,1)}
            est = linear_model.LogisticRegression(penalty = 'l1',class_weight = 'auto')
            est = grid_search.GridSearchCV(est, parameters)
            est = est.fit(train_X_data,train_Y_data)

            print("Best parameters set found on development set:")
            print()
            print(est.best_estimator_)            
        else:
            est = linear_model.LogisticRegression(penalty = 'l1',class_weight = 'auto')
            est = est.fit(train_X_data,train_Y_data)

    print ('Predicting...')
    prd_Y_data = est.predict(test_data).astype(int)

    predictions_file = open("EstimationResult.csv", "w",newline='')
    open_file_object = csv.writer(predictions_file)
    open_file_object.writerow(["id","Prd_click","Tr_click"])
    open_file_object.writerows(zip(ids,prd_Y_data,test_Y_data))
    predictions_file.close()
    #print ('T_size is %s' % len(test_Y_data))

    #evaluates crossValidation Score
    get_Kfold_CrossValidation_score(train_X_data, train_Y_data,est,3)

    #evaluates prediction
    get_evaluation(test_Y_data,prd_Y_data,[True,True,True,False,False])

print ('Done...')
    


