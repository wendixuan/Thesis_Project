###################################
#
#
#
# Code for experiements for ANOVA kernel
#
#
#
####################################
import numpy as np
from sys import path
path.append('./project')
import anova as kernels
from sklearn import cross_validation
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
import time

#Load and reshape data set
X=np.genfromtxt('./movement_libras',delimiter=',')
XT=np.dstack((X[:,0:-1:2],X[:,1:-1:2]))
XT=XT.transpose((0,2,1))
Xflatd=kernels.DataTabulator(XT)
labels=X[:,-1]
a=0.4796037030716088
tuning_grid = [
    {'svc__C': [1,100,10000], 'AnovaKernelizer__subsample': [1,5],'AnovaKernelizer__scale': [0.2*a,0.5*a,a,2*a,5*a]}
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