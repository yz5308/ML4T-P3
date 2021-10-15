""""""
"""  		  	   		   	 		  		  		    	 		 		   		 		  
Test a learner.  (c) 2015 Tucker Balch  		  	   		   	 		  		  		    	 		 		   		 		  
  		  	   		   	 		  		  		    	 		 		   		 		  
Copyright 2018, Georgia Institute of Technology (Georgia Tech)  		  	   		   	 		  		  		    	 		 		   		 		  
Atlanta, Georgia 30332  		  	   		   	 		  		  		    	 		 		   		 		  
All Rights Reserved  		  	   		   	 		  		  		    	 		 		   		 		  
  		  	   		   	 		  		  		    	 		 		   		 		  
Template code for CS 4646/7646  		  	   		   	 		  		  		    	 		 		   		 		  
  		  	   		   	 		  		  		    	 		 		   		 		  
Georgia Tech asserts copyright ownership of this template and all derivative  		  	   		   	 		  		  		    	 		 		   		 		  
works, including solutions to the projects assigned in this course. Students  		  	   		   	 		  		  		    	 		 		   		 		  
and other users of this template code are advised not to share it with others  		  	   		   	 		  		  		    	 		 		   		 		  
or to make it available on publicly viewable websites including repositories  		  	   		   	 		  		  		    	 		 		   		 		  
such as github and gitlab.  This copyright statement should not be removed  		  	   		   	 		  		  		    	 		 		   		 		  
or edited.  		  	   		   	 		  		  		    	 		 		   		 		  
  		  	   		   	 		  		  		    	 		 		   		 		  
We do grant permission to share solutions privately with non-students such  		  	   		   	 		  		  		    	 		 		   		 		  
as potential employers. However, sharing with other current or future  		  	   		   	 		  		  		    	 		 		   		 		  
students of CS 7646 is prohibited and subject to being investigated as a  		  	   		   	 		  		  		    	 		 		   		 		  
GT honor code violation.  		  	   		   	 		  		  		    	 		 		   		 		  
  		  	   		   	 		  		  		    	 		 		   		 		  
-----do not edit anything above this line---  		  	   		   	 		  		  		    	 		 		   		 		  
"""

import math
import sys

import numpy as np

import LinRegLearner as lrl
import DTLearner as dt
import RTLearner as rt
import BagLearner as bl
import InsaneLearner as it

import matplotlib.pyplot as plt
import matplotlib
import time
                          
if __name__=="__main__":

    if len(sys.argv) != 2:                          
        print("Usage: python testlearner.py <filename>")                          
        sys.exit(1)

    if sys.argv[1] == "Data/Istanbul.csv":
        inf = open(sys.argv[1])
        data = np.genfromtxt(inf,delimiter=',')
        data = data[1:,1:]
    else:
        inf = open(sys.argv[1])
        data = np.array(list([map(float,s.strip().split(',')) for s in inf.readlines()]))

    # print (data[:,0:-1].shape)
    # print(data[:, -1].shape)

    # compute how much of the data is training and testing
    train_rows = int(0.4 * data.shape[0])
    test_rows = data.shape[0] - train_rows

    # separate out training and testing data
    train_x = data[:train_rows, 0:-1]
    train_y = data[:train_rows, -1]
    test_x = data[train_rows:, 0:-1]
    test_y = data[train_rows:, -1]

    # print(f"{test_x.shape}")
    # print(f"{test_y.shape}")


    # create a learner and train it
    learner = lrl.LinRegLearner(verbose=True)  # create a LinRegLearner
    learner.add_evidence(train_x, train_y)  # train it
    print(learner.author())


    ###Experiment 1
    dtTrainRMSE = []
    dtTestRMSE = []
    for i in range(1,51):
        learner = dt.DTLearner(leaf_size=i,verbose = True)
        learner.add_evidence(train_x, train_y)

        # evaluate in sample
        predY = learner.query(train_x)  # get the predictions
        rmse = math.sqrt(((train_y - predY) ** 2).sum() / train_y.shape[0])
        dtTrainRMSE.append(rmse)
        print()
        print("In sample results")
        print(f"RMSE: {rmse}")
        c = np.corrcoef(predY, y=train_y)
        print(f"corr: {c[0, 1]}")

        # evaluate out of sample
        predY = learner.query(test_x)  # get the predictions
        rmse = math.sqrt(((test_y - predY) ** 2).sum() / test_y.shape[0])
        dtTestRMSE.append(rmse)
        print()
        print("Out of sample results")
        print(f"RMSE: {rmse}")
        c = np.corrcoef(predY, y=test_y)
        print(f"corr: {c[0, 1]}")


    plt.figure(1)
    plt.plot(range(1,51), dtTrainRMSE,  label='Train: In_Sample')
    plt.plot(range(1,51), dtTestRMSE, label='Test: Out_Sample')
    plt.title("RMSE vs Leaf Size: DTLearner")
    plt.xlabel("Leaf Size")
    plt.ylabel("RMSE")
    plt.legend(loc='best')
    plt.savefig("Q1-Dtlearner.png")

    # compute how much of the data is training and testing                          
    train_rows = int(0.6* data.shape[0])
    test_rows = data.shape[0] - train_rows                          
                          
    # separate out training and testing data                          
    train_x = data[:train_rows,0:-1]                          
    train_y = data[:train_rows,-1]                          
    test_x = data[train_rows:,0:-1]                          
    test_y = data[train_rows:,-1]
    
    
    ##Experiment 2
    BagTrainRMSE = []
    BagTestRMSE = []
    for i in range(1, 51):
        learner = bl.BagLearner(learner=dt.DTLearner, kwargs={"leaf_size": i}, bags=20, boost=False, verbose=False)
        learner.add_evidence(train_x, train_y)
        predY = learner.query(train_x)
        BagTrainRMSE.append(np.sqrt(np.mean((predY - train_y) ** 2)))
        predY = learner.query(test_x)
        BagTestRMSE.append(np.sqrt(np.mean((predY - test_y) ** 2)))

    plt.figure(2)
    plt.plot(range(1,51), BagTrainRMSE,  label='Train: In_Sample')
    plt.plot(range(1,51), BagTestRMSE, label='Test: Out_Sample')
    plt.title("RMSE vs Leaf Size: Bag Learner")
    plt.xlabel("Leaf Size")
    plt.ylabel("RMSE")
    plt.legend(loc='best')
    plt.savefig("Q2-Baglearner.png")

    ###Experiment 3: tree depth & build time

    dt_out_sample_error_mean = []
    rt_out_sample_error_mean = []
    dtBuildTime = []
    rtBuildTime = []
    dtDepth = []
    rtDepth = []


    for i in range(1, 51):
        start = time.time()
        learner = dt.DTLearner(leaf_size=i, verbose=True)
        learner.add_evidence(train_x, train_y)
        end = time.time()
        dtBuildTime.append(end - start)
        dtDepth.append(int(np.log2(learner.num_leafs())))


        predY = learner.query(test_x)
        dt_test_mean_error = abs(test_y - predY).sum()/test_y.shape[0]
        dt_out_sample_error_mean.append(dt_test_mean_error)

        start = time.time()
        learner = rt.RTLearner(leaf_size=i, verbose=True)
        learner.add_evidence(train_x, train_y)
        end = time.time()
        rtBuildTime.append(end - start)
        rtDepth.append(int(np.log2(learner.num_leafs())))

        predY = learner.query(test_x)
        rt_test_mean_error = abs(test_y - predY).sum()/test_y.shape[0]
        rt_out_sample_error_mean.append(rt_test_mean_error)


    print (dt_out_sample_error_mean, rt_out_sample_error_mean)
    print (dtBuildTime, rtBuildTime)
    print (dtDepth, rtDepth)


    plt.figure(3)
    plt.plot(range(1,51), dt_out_sample_error_mean,  label='DTLEarner')
    plt.plot(range(1,51), rt_out_sample_error_mean, label='RTLearner')
    plt.title("Out of Sample Mean Absolute Error : DTLearner vs RTLearner")
    plt.xlabel("Leaf Size")
    plt.ylabel("Mean Absolute Error")
    plt.legend(loc='best')
    plt.savefig("Q3-MeanAbsoluteError DT&RT.png")


    plt.figure(4)
    plt.plot(range(1,51), dtBuildTime,  label='DTLEarner')
    plt.plot(range(1,51), rtBuildTime, label='RTLearner')
    plt.title("Time to Build Tree: DTLearner vs RTLearner")
    plt.xlabel("Leaf Size")
    plt.ylabel("Time (s)")
    plt.legend(loc='best')
    plt.savefig("Q3-BuildTime DT&RT.png")


    plt.figure(5)
    plt.plot(range(1,51), dtDepth,  label='DTLEarner')
    plt.plot(range(1,51), rtDepth, label='RTLearner')
    plt.title("Average Tree Depth : DTLearner vs RTLearner")
    plt.xlabel("Leaf Size")
    plt.ylabel("Depth Level(Log of number of leaves)")
    plt.legend(loc='best')
    plt.savefig("Q3-TreeDepth DT&RT.png")


