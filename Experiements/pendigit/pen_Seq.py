###################################
#
#
#
# Code for experiements for sequential kernel
#
#
#
####################################
from sys import path
path.append('./')
import numpy as np
from sklearn import cross_validation
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
import kernels as kernels
import time

#Load and reshape data set
#load data
X=np.genfromtxt('./pendigits',delimiter=',')
Xflatd=X[:,:16]
labels=X[:,-1]

#Seq of kernel with original
a=203.56817039999154
tuning_grid = [
    {'svc__C': [1,100,10000], 'SeqKernelizer__kernel': ['GA'], 
    'SeqKernelizer__Level': [2,3,4],
     'SeqKernelizer__subsample': [1],'SeqKernelizer__scale': [0.2*a,0.5*a,a,2*a,5*a]}
]
Seqpip= GridSearchCV(kernels.SeqSVCpipeline, tuning_grid, cv = 3,scoring='accuracy')


start_Seq=time.clock()
scores_Seq = cross_validation.cross_val_score(Seqpip, Xflatd, labels, cv=3,scoring='accuracy')
end_Seq=time.clock()

print 'Seq of kernel'
print tuning_grid
print ("Mean: %0.2f " % (scores_Seq.mean()))
print 'Time:'
print end_Seq-start_Seq

