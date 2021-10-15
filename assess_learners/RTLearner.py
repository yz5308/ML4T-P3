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
import random
                          
class RTLearner(object):

    def __init__(self, leaf_size=1, verbose=True):
        # define a global variable leaf_size since decision tree has leaf size feature, while LinReg doesn’t have
        self.leaf_size = leaf_size  # Add leaf_size feature to learner

    def author(self):
        return 'yzhao633'  # replace tb34 with your Georgia Tech username

    def add_evidence(self, dataX, dataY):
        """
        @summary: Add training data to learner
        @param dataX: X values of data to add
        @param dataY: the Y training values
        """

        # slap on 1s column so linear regression finds a constant term
        newdataX = np.ones([dataX.shape[0], dataX.shape[1] + 1])
        newdataX[:, 0:dataX.shape[1]] = dataX

        # row = row of dataX, column = column of dataX + 1, extra column is for dataY
        newdataX[:, -1] = dataY  # combine dateX and dateY into newdataX
        # apply sub function build_tree
        self.tree = self.build_tree(newdataX)

    def build_tree(self, data):
        # if data.shape[0] == 1: return [leaf, data.y, NA, NA]
        # while numbers of data is less than the defined left_size, it is not applicable, return NA
        if data.shape[0] <= self.leaf_size:
            return np.array([[-1, data[0][-1], -1, -1]])
        # if all data.y same: return [leaf, data.y, NA, NA]
        # while all dataY are the same
        elif np.all(data[:, -1] == data[0, -1]):
            return np.array([[-1, data[0][-1], -1, -1]])
        # else
        # determine best feature i to split on
        # SplitVal = data[:,i].median()
        # lefttree = build_tree(data[data[:,i]<=SplitVal])
        # righttree = build_tree(data[data[:,i]>SplitVal])
        # root = [i, SplitVal, 1, lefttree.shape[0] + 1]
        # return (append(root, lefttree, righttree))
        else:
            ## I used the same method as the Video - Using Correlation to determine “best” feature
            # row #	X2	X10	X11	 Y
            # Correl -0.731  0.406	0.826
            # X11 has highest correlation
            corr = []
            # calculate and record the correlation of i and and dataY
            for i in range(data.shape[1] - 1):
                col = data[:, i]
                corr.append(np.corrcoef(col, data[:, -1])[0, 1])
            corr = np.absolute(corr)  # Corr =  [0.731 ,0.406, 0.826]
            # Take the i that has highest correlation with dataY as the feature
            feature = np.argmax(corr)  # x11’index
            # to make the tree balanced, use the median value of the feature as the split value
            split_val = np.median(data[:, feature], axis=0)

            # if the feature's median = feature's max, like [1,2,2,2] it is not applicable, return NA
            if (split_val == np.max(data[:, feature], axis=0)):
                temp = np.argmax(data[:, feature])
                return np.array([[-1, data[temp][-1], -1, -1]])

            # Use recursion to build the tree,
            lefttree = self.build_tree(data[data[:, feature] <= split_val])
            righttree = self.build_tree(data[data[:, feature] > split_val])
            ##  [festure, split_val, index of root.left, index of root right]
            root = np.array([[feature, split_val, 1, lefttree.shape[0] + 1]])  # X11’s index   9.900  1 7

            # Return the Preorder Traversal
            return np.vstack((root, lefttree, righttree))

    def query(self, points):
        """
        @summary: Estimate a set of test points given the model we built.
        @param points: should be a numpy array with each row corresponding to a specific query.
        @returns the estimated values according to the saved model.
        """
        result = []
        # since there are more than one point, we can append the result one by one
        for point in points:
            result.append(self.query_tree(point, node=0))
        return np.array(result)  ## result can be an array

    def query_tree(self, point, node):
        ## As for query trees, there are only 3 possibilities, and I discussed those one by one
        node_index = self.tree[node]  # self.tree[0] = [X11’s index  9.900  1   7]
        current_index_int = int(node_index[0])  # X11’s index
        if current_index_int == -1:
            return node_index[1]  # 9.900
        elif point[current_index_int] > node_index[1]:  # point[X11’s indes] > 9.900, move to right tree
            new_index = node + node_index[3]  # node’s index + index of node right
        else:  # point[X11] <= 9.900, move to left tree
            new_index = node + node_index[2]  # node’s index + index of node left
        return self.query_tree(point,
                               int(new_index))  # recursion until node_index[0] == -1 and node_index[1]is returned

    def num_leafs(self):
        leafs = 0
        # to calculate the number of end leafs, since end leafs have no feature anymore, it is -1
        # this function is used for experiment 3 :
        # for balanced tree: depth*depth = number of end leafs
        for i in range(self.tree.shape[0]):
            if self.tree[i][0] == -1:
                leafs = leafs + 1
        return leafs


if __name__ == "__main__":
    print("the secret clue is 'zzyzx'")

    # """
# row #X2	    X10	    X11	   Y
# cor 1.000	-1.000	1.000
# 4	0.610	0.630	8.400	3.000	Left Left
# 0	0.885	0.330	9.100	4.000
# 	-1.000	-1.000	-1.000
# 2	0.560	0.500	9.400	6.000	Left Right
# 3	0.735	0.570	9.800	5.000
# 	-1.000	-1.000	1.000
# 7	0.320	0.780	10.000	6.000	Right Left
# 5	0.260	0.630	11.800	8.000
# 	-1.000	1.000	-1.000
# 6	0.500	0.680	10.500	7.000	Right Right
# 1	0.725	0.390	10.900	5.000
#
# Tree
# node Factor SplitVal Left Right
# 0	 X11	 9.900	1   7
# 1	 X11	 9.250	1	4
# 2   X2	 0.748	1	2
# 3  	Leaf	3.000	NA	NA
# 4	Leaf	4.000	NA	NA
# 5	X2	0.648	1	2
# 6	Leaf	6.000	NA	NA
# 7	Leaf	5.000	NA	NA
# 8	X2	0.410	1	4
# 9	X11	10.900	1	2
# 10	Leaf	6.000	NA	NA
# 11	Leaf	8.000	NA	NA
# 12	X11	10.700	1	2
# 13	Leaf	7.000	NA	NA
# 14	Leaf	5.000	NA	NA
#
