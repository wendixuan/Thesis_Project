
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
path.append('./project')
import kernels as kernels
from sklearn import cross_validation
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
import glob
import time

#Input data
d={}
f1=glob.glob('./AUSLAN/tctodd1/*.tsd')
f2=glob.glob('./AUSLAN/tctodd2/*.tsd')
f3=glob.glob('./AUSLAN/tctodd3/*.tsd')
f4=glob.glob('./AUSLAN/tctodd4/*.tsd')
f5=glob.glob('./AUSLAN/tctodd5/*.tsd')
f6=glob.glob('./AUSLAN/tctodd6/*.tsd')
f7=glob.glob('./AUSLAN/tctodd7/*.tsd')
f8=glob.glob('./AUSLAN/tctodd8/*.tsd')
f9=glob.glob('./AUSLAN/tctodd9/*.tsd')
f=f1+f2+f3+f4+f5+f6+f7+f8+f9
for r in f:
    t=np.genfromtxt(r)
    d[r]=t
    
k=list(d.keys())
k.sort()  
for l in k:
    d[l]=d[l].T
#Tranform data to be 3D array, and record the lengths of  time series 
def DictTo2Darray(d,orderlist,maxnum,numfeatures):
    X=np.zeros((len(d),numfeatures,maxnum))
    V=np.zeros(len(d))
    t=0
    for l in orderlist:
        shape=d[l].shape
        V[t]=shape[1]
        X[t,:,:shape[1]]=d[l]
        t=t+1
    return X,V
X,V=DictTo2Darray(d,k,136,22)
X.shape
Xflatd1=kernels.DataTabulator(X)[:,:]
V1=V[:]
labels1=np.tile(np.repeat(np.array(range(95))+1,3),9)[:]
Xflatd2=np.insert(Xflatd1, -1, V1.T, axis=1)
Xflatd=Xflatd2[:,:]
labels=labels1[:]
#Seq of kernel 
#Experiements
#Sumsample interval=10
a=1.798773145134205
tuning_grid = [
    {'svc__C': [1,100,10000], 'SeqKernelizer__kernel': ['GA'], 'SeqKernelizer__differences': [True],
    'SeqKernelizer__Level': [2,3,4],'SeqKernelizer__numfeatures':[22],'SeqKernelizer__V':[True],
     'SeqKernelizer__subsample': [5,10],'SeqKernelizer__scale': [0.2*a,0.5*a,a,2*a,5*a]}
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



