###################################
#
#
#
# Code for experiements for s-sequential kernel
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

#S-sequential kernel
a=203.56817039999154
tuning_grid = [
    {'svc__C': [1,100,10000], 'SeqKernelizer__kernel': ['GA'], 
    'SeqKernelizer__Level': [2,3,4],'SeqKernelizer__Dia':[True],
     'SeqKernelizer__subsample': [1],'SeqKernelizer__scale': [0.2*a,0.5*a,a,2*a,5*a]}
]
SSpip= GridSearchCV(kernels.SeqSVCpipeline, tuning_grid, cv = 3,scoring='accuracy')

start_SS=time.clock()
scores_SS = cross_validation.cross_val_score(SSpip, Xflatd, labels, cv=3,scoring='accuracy')
end_SS=time.clock()

print 'S-sequential kernel '
print tuning_grid
print ("Mean: %0.2f " % (scores_SS.mean()))
print 'Time:'
print end_SS-start_SS

