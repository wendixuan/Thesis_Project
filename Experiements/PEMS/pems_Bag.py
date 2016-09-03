###################################
#
#
#
# Code for experiements for bag of features kernel
#
#
#
####################################
import numpy as np
import scipy.io as sio
from sys import path
path.append('./')
import kernels as kernels
from sklearn import cross_validation
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
import time

#Load and reshape data set
train=sio.loadmat('./pems.mat')
X=train['pems']
labels= np.genfromtxt('./pemslabels.txt',delimiter=',')

N = np.shape(labels)[0]
# reshape to 2D array for compatibility with scikit-learn.
#  Internally reshaped back to 3D array in estimator
Xflatd = kernels.DataTabulator(X)

#Bag of kernel
a=4.986301133104573
tuning_grid = [
    {'svc__C': [1,100,10000], 'BagKernelizer__kernel': ['GA'], 'BagKernelizer__differences': [True],
     'BagKernelizer__numfeatures':[963],
     'BagKernelizer__subsample': [10,20],'BagKernelizer__scale': [0.2*a,0.5*a,a,2*a,5*a]}
]
Bagpip= GridSearchCV(kernels.BagSVCpipeline, tuning_grid, cv = 3,scoring='accuracy')


start_Bag=time.clock()
scores_Bag = cross_validation.cross_val_score(Bagpip, Xflatd, labels, cv=10,scoring='accuracy')
end_Bag=time.clock()

print 'Bag of kernel:'
print tuning_grid
print ("Mean: %0.2f " % (scores_Bag.mean()))
print 'Time:'
print end_Bag-start_Bag
