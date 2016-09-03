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
from sys import path
path.append('./')
import kernels as kernels
from sklearn import cross_validation
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
import time

#Load and reshape data set
X=np.genfromtxt('./movement_libras',delimiter=',')
XT=np.dstack((X[:,0:-1:2],X[:,1:-1:2]))
XT=XT.transpose((0,2,1))
Xflatd=Xflatd = kernels.DataTabulator(XT)
labels=X[:,-1]

#Sequetial kernel 
a=0.25375937618145222
tuning_grid = [
    {'svc__C': [1,100,10000], 'SeqKernelizer__kernel': ['GA'], 
    'SeqKernelizer__Level': [2,3,4],
     'SeqKernelizer__subsample': [1,5],'SeqKernelizer__scale': [0.2*a,0.5*a,a,2*a,5*a]}
]

Seqpip= GridSearchCV(kernels.SeqSVCpipeline, tuning_grid, cv = 3,scoring='accuracy')

start_Seq=time.clock()
scores_Seq = cross_validation.cross_val_score(Seqpip, Xflatd, labels, cv=3,scoring='accuracy')
end_Seq=time.clock()

print 'Seq of kernel--change'
print tuning_grid
print ("Mean: %0.2f " % (scores_Seq.mean()))
print 'Time:'
print end_Seq-start_Seq


