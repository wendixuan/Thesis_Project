###################################
#
#
#
# Code for experiements for ANOVA kernel
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
import anova as kernels
import time
#Load and reshape data set
#load data
X=np.genfromtxt('./pendigits',delimiter=',')
Xflatd=X[::7,:16]
labels=X[::7,-1]


a=203.56817039999154
tuning_grid = [
    {'svc__C': [1,100,10000], 'AnovaKernelizer__numfeatures': [2],
     'AnovaKernelizer__subsample': [1],'AnovaKernelizer__scale': [0.2*a,0.5*a,a,2*a,5*a]}
]
Anovapip= GridSearchCV(kernels.AnovaSVCpipeline, tuning_grid, cv = 3)

start_Anova=time.clock()
scores_Anova = cross_validation.cross_val_score(Anovapip, Xflatd, labels, cv=3)
end_Anova=time.clock()

print 'Anova of kernel--change'
print tuning_grid
print ("Mean: %0.2f " % (scores_Anova.mean()))
print 'Time:'
print end_Anova-start_Anova