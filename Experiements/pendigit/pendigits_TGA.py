import numpy as np
from sys import path
path.append('./')
import tgakernel as kernels
from sklearn import cross_validation
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
import time

#Load and reshape data set
#load data
X=np.genfromtxt('./pendigits',delimiter=',')
Xflatd=X[:,:16]
labels=X[:,-1]
a=203.56817039999154
#Seq of kernel with original
tuning_grid = [
   {'svc__C': [1,100,10000],  
    'TGAKernelizer__sigma': [0.2*a,0.5*a,a,2*a,5*a],
    'TGAKernelizer__triangular': [2,4] }
]
TGApip= GridSearchCV(kernels.TGASVCpipeline, tuning_grid, cv = 3)

start_TGA=time.clock()
scores_TGA = cross_validation.cross_val_score(TGApip, Xflatd, labels, cv=3)
end_TGA=time.clock()

print 'TGA of kernel'
print tuning_grid
print ("Mean: %0.2f " % (scores_TGA.mean()))
print 'Time:'
print end_TGA-start_TGA
