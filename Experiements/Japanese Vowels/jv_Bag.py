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
file=open('./ae')
t=0
d={}
tmp=[]
for line in file:
    if (len(line)==1 or line==[]):
        d[t]=np.array(tmp)
        tmp=[]
        t=t+1
    else:
        l_tmp= [float(i) for i in line.split()] 
        tmp.append(l_tmp)


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
X,V=DictTo2Darray(d,k,29,12)
X.shape
Xflatd1=kernels.DataTabulator(X)

labels1=np.repeat(np.array(range(9))+1,30)
Xflatd=np.insert(Xflatd1, -1, V.T, axis=1)
labels=labels1[:]

#Bag kernel

#Bag kernel
a=1.1399008014156318
tuning_grid = [
    {'svc__C': [1,100,10000], 'BagKernelizer__kernel': ['GA'], 'BagKernelizer__differences': [True],
     'BagKernelizer__numfeatures':[12],'BagKernelizer__V':[True],
     'BagKernelizer__subsample': [1,4],'BagKernelizer__scale': [0.2*a,0.5*a,a,2*a,5*a]}
]
Bagpip= GridSearchCV(kernels.BagSVCpipeline, tuning_grid, cv = 3,scoring='accuracy')


start_Bag=time.clock()
scores_Bag = cross_validation.cross_val_score(Bagpip, Xflatd, labels, cv=3,scoring='accuracy')
end_Bag=time.clock()
# In[]
print 'Bag of kernel:--change '
print tuning_grid
print ("Mean: %0.2f " % (scores_Bag.mean()))
print 'Time:'
print end_Bag-start_Bag
