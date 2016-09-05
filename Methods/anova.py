
import numpy as np

#function for a distance between two path
def sqdist(X,Y):
    M = np.shape(X)[0]
    N = np.shape(Y)[0]
    return np.tile((X*X).sum(-1),[N,1]).T + np.tile((Y*Y).sum(-1),[M,1]) - 2*np.inner(X,Y)
#Reshape the dataset to be 2D
def DataTabulator(X):
    
    Xshape = np.shape(X)
        
    return np.reshape(X,(Xshape[0],np.prod(Xshape[1:])))
#Reshape dataset to be 3D
def TimeSeriesReshaper(Xflat, numfeatures, subsample = 1, differences = True):
    flatXshape = np.shape(Xflat)
    Xshape = (flatXshape[0],numfeatures,flatXshape[1]/numfeatures)        
    X = np.reshape(Xflat,Xshape)[:,:,::subsample]
    
    if differences:
        return np.diff(X)
    else:    
        return X
#a base kernel of ANOVA kenerl 
def kGA(x,y,scale): 
    return np.exp(-(sqdist(x,y)/(2*(scale**2))+np.log(2-np.exp(-sqdist(x,y)/(2*(scale**2))))))
    
#compute a ANOVA kernel of degree n
def anova(X,Y,scale):
    K=kGA(X,Y,scale)
    return sum(K.diagonal().cumprod())

##Compute cross-matrix
def AnovaKernelXY(X,Y,scale):

    N = np.shape(X)[0]   
    M = np.shape(Y)[0]   
    
    KSeq = np.zeros((N,M))
        
    for row1ind in range(N):
        for row2ind in range(M):
            KSeq[row1ind,row2ind]= anova(X[row1ind].T,Y[row2ind].T,scale)
          
    
    return KSeq

from sklearn.base import BaseEstimator, TransformerMixin
#Class ANOAkernelizer

#    scale= scaling constant of the primary kernel funtion
#    numfeatures = number of features per time point, for internal reshaping
#    subsample = time series is subsampled to every subsample-th time point
#    differences = whether first differences are taken or not
 

class AnovaKernelizer(BaseEstimator, TransformerMixin):
    def __init__(self, X = np.zeros((1,2)), scale=1,
                 numfeatures = 2, subsample = 1, differences =True
                ):
        self.scale=scale
        self.subsample= subsample
        self.numfeatures=numfeatures
        self.differences=differences
        
        self.X = X
        
    def fit(self, X, y=None):
        self.X = TimeSeriesReshaper(X,self.numfeatures,self.subsample,self.differences)
        return self
        
    def transform(self,Y):
        Y = TimeSeriesReshaper(Y,self.numfeatures,self.subsample,self.differences)
        KSeq = AnovaKernelXY(Y,self.X,self.scale)
        return KSeq

from sklearn import svm
from sklearn.pipeline import Pipeline
##pipeline:Anova kernel with SVC
AnovaSVCpipeline=Pipeline([

    ('AnovaKernelizer', AnovaKernelizer()),
    
    ('svc', svm.SVC(kernel = 'precomputed'))

])





















