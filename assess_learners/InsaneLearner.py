import numpy as np
import LinRegLearner as lrl
import BagLearner as bl
class InsaneLearner(object):
   def __init__(self, leaf_size = 1, verbose = True):
## InsaneLearner should contain 20 BagLearner instances where each instance is composed of 20 LinRegLearner instances.
       for i in range(20):
           self.learner = bl.BagLearner(learner=lrl.LinRegLearner, kwargs={}, bags=20, boost= False, verbose= False)
   def author(self):
       return 'yzhao633' # replace tb34 with your Georgia Tech username
   def add_evidence(self,dataX,dataY):
## InsaneLearner should contain 20 BagLearner instances where each instance is composed of 20 LinRegLearner instances.
       for i in range(20):
           self.learner.add_evidence(dataX,dataY)
   def query(self,points):
## These code are similar as DT’s query function, the difference is the value return is the mean of
## the y_test instead of array of y_test itself
       q = []
       for i in range(20):
           q.append(self.learner.query(points))
       return np.mean(q,axis=0)
if __name__=="__main__":
   print("the secret clue is 'zzyzx'")

