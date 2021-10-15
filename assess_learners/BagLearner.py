"""                          
A simple wrapper for linear regression.  (c) 2015 Tucker Balch                          
                          
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

import numpy as np
import DTLearner as dt
import RTLearner as rt
import random


class BagLearner(object):

    ## Based on given example, I defined learner, kwargs, bags, boost and verbose

    def __init__(self, learner=dt.DTLearner, kwargs={}, bags=20, boost=False, verbose=False):
        self.learner = learner
        self.kwargs = kwargs
        self.bags = bags
        self.boost = boost
        self.verbose = verbose
        # learners has 20 learner, this is the feature of Bag Learner, it has 20 bags
        self.learners = [learner(**kwargs) for i in range(bags)]  # use ** to unpack dictionary;

    def author(self):
        return 'yzhao633'  # replace tb34 with your Georgia Tech username

    def getData(self, Xtrain, Ytrain):
        # randomly select list of nums with numbers of Ytrain with replacement
        nums = np.random.choice(Ytrain.shape[0], Ytrain.shape[0], replace=True)
        return Xtrain[nums], Ytrain[nums]

    def add_evidence(self, Xtrain, Ytrain):
        """
        @summary: Add training data to learner
        @param dataX: X values of data to add
        @param dataY: the Y training values
        """
        # apply random data to 20 learners in self.learners one by one.
        for learner in self.learners:
            xData, yData = self.getData(Xtrain, Ytrain)
            learner.add_evidence(xData, yData)  # learner = dt.DTLearner

    def query(self, xTest):
        """
        @summary: Estimate a set of test points given the model we built.
        @param points: should be a numpy array with each row corresponding to a specific query.
        @returns the estimated values according to the saved model.
        """

        ## These code are similar as DT’s query function, the difference is the value return is the mean of
        ## the y_test instead of array of y_test itself
        result = 0  # or: result = [] if we want to get all the predict yTest
        for learner in self.learners:
            result += learner.query(xTest)  ## or : result.append(learner.query(xTest))
        # calculate the average of Predicted yTest
        result = result / len(self.learners)  ## or : result = result.mean()
        return result


if __name__ == "__main__":
    print("the secret clue is 'zzyzx'")









