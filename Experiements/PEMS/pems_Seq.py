###################################
#
#
#
# Code for experiements for sequential kernel
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


#Sequential kernel
a=4.986301133104573
tuning_grid = [
    {'svc__C': [1,100,10000], 'SeqKernelizer__kernel': ['GA'], 'SeqKernelizer__differences': [True],
    'SeqKernelizer__Level': [2,3,4],'SeqKernelizer__numfeatures':[963],
     'SeqKernelizer__subsample': [10,20],'SeqKernelizer__scale': [0.2*a,0.5*a,a,2*a,5*a]}
]
Seqpip= GridSearchCV(kernels.SeqSVCpipeline, tuning_grid, cv = 3,scoring='accuracy')

start_Seq=time.clock()
scores_Seq = cross_validation.cross_val_score(Seqpip, Xflatd, labels, cv=3,scoring='accuracy')
end_Seq=time.clock()

print 'Sequential kernel:'
print tuning_grid
print ("Mean: %0.2f " % (scores_Seq.mean()))
print 'Time:'
print end_Seq-start_Seq
