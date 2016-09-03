
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

#Bag kernel

#Experiments

a=0.4796037030716088
tuning_grid = [
    {'svc__C': [1,100,10000], 'BagKernelizer__kernel': ['GA'], 
     'BagKernelizer__subsample': [1,5],'BagKernelizer__scale': [0.2*a,0.5*a,a,2*a,5*a]}
]
Bagpip= GridSearchCV(kernels.BagSVCpipeline, tuning_grid, cv = 3)

start_Bag=time.clock()
scores_Bag = cross_validation.cross_val_score(Bagpip, Xflatd, labels, cv=3,scoring='accuracy')
end_Bag=time.clock()

print 'Bag of kernel:'
print tuning_grid
print ("Mean: %0.2f " % (scores_Bag.mean()))
print 'Time:'
print end_Bag-start_Bag
