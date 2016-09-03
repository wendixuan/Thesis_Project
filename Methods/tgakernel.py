import global_align as ga
import numpy as np
from sklearn import preprocessing
def DataTabulator(X):
    
    Xshape = np.shape(X)
        
    return np.reshape(X,(Xshape[0],np.prod(Xshape[1:])))

def TimeSeriesReshaper(Xflat, numfeatures, subsample = 1, differences = False):
    flatXshape = np.shape(Xflat)
    Xshape = (flatXshape[0],numfeatures,flatXshape[1]/numfeatures)        
    X = np.reshape(Xflat,Xshape)[:,:,::subsample]
    #X1=preprocessing.Normalizer(X).norm
    if differences:
        return np.diff(X)
    else:    
        return X



##Compute cross-matrix
def TGAKernelXY(X,Y,sigma=0.1,triangular=10):

    N = np.shape(X)[0]   
    M = np.shape(Y)[0]   
    
    KSeq = np.zeros((N,M))
        
    for row1ind in range(N):
        for row2ind in range(M):
            val = ga.tga_dissimilarity(X[row1ind].T.copy(order='C'),Y[row2ind].T.copy(order='C'),sigma, triangular)
           # KSeq[row1ind,row2ind]=val
            KSeq[row1ind,row2ind]=np.exp(-val)
            
    return KSeq

from sklearn.base import BaseEstimator, TransformerMixin
#Class Bagkernelizer
class TGAKernelizer(BaseEstimator, TransformerMixin):
    def __init__(self,sigma=0.1,triangular=0.25, X = np.zeros((1,2)), 
                 numfeatures = 2, subsample = 1, differences =False
                ):
        self.sigma=sigma
        self.subsample= subsample
        self.triangular=triangular
        self.numfeatures=numfeatures
        self.differences=differences
        self.X = X
        
    def fit(self, X, y=None):
        self.X = TimeSeriesReshaper(X,self.numfeatures,self.subsample,self.differences)
        return self
        
    def transform(self,Y):
        Y = TimeSeriesReshaper(Y,self.numfeatures,self.subsample,self.differences)
        KSeq = TGAKernelXY(Y,self.X,self.sigma,self.triangular)
        return KSeq

from sklearn import svm
from sklearn.pipeline import Pipeline

##pipeline:TGA kernel with SVC
TGASVCpipeline=Pipeline([ 

    ('TGAKernelizer', TGAKernelizer()),
    
    ('svc', svm.SVC(kernel = 'precomputed'))

])





















