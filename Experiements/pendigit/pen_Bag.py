###################################
#
#
#
# Code for experiements for bag of features kernel
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
X=np.genfromtxt('./pendigits',delimiter=',')
Xflatd=X[:,:16]
labels=X[:,-1]

#Seq of kernel with original
a=203.56817039999154
tuning_grid = [
    {'svc__C': [1,100,10000], 'BagKernelizer__kernel': ['GA'], 
     'BagKernelizer__subsample': [1],'BagKernelizer__scale': [0.2*a,0.5*a,a,2*a,5*a]}
]
Bagpip= GridSearchCV(kernels.BagSVCpipeline, tuning_grid, cv = 3,scoring='accuracy')


start_Bag=time.clock()
scores_Bag= cross_validation.cross_val_score(Bagpip, Xflatd, labels, cv=3,scoring='accuracy')

end_Bag=time.clock()

print 'Bag of kernel'
print tuning_grid
print ("Mean: %0.2f " % (scores_Bag.mean()))
print 'Time:'
print end_Bag-start_Bag

