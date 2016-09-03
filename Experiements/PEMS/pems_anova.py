import numpy as np
import scipy.io as sio
from sys import path
path.append('./')
import anova as kernels
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



a=4.986301133104573
tuning_grid = [
    {'svc__C': [1,10,10000],  'AnovaKernelizer__differences': [True],'AnovaKernelizer__numfeatures':[963],
     'AnovaKernelizer__subsample': [10,20],'AnovaKernelizer__scale': [0.2*a,0.5*a,a,2*a,5*a]}
]
Anovapip= GridSearchCV(kernels.AnovaSVCpipeline, tuning_grid, cv = 3)

start_Anova=time.clock()
scores_Anova = cross_validation.cross_val_score(Anovapip, Xflatd, labels, cv=3)
end_Anova=time.clock()

print 'Anova of kernel'
print tuning_grid
print ("Mean: %0.2f " % (scores_Anova.mean()))
print 'Time:'
print end_Anova-start_Anova