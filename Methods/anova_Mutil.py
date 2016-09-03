
import numpy as np


def sqdist(X,Y):
    M = np.shape(X)[0]
    N = np.shape(Y)[0]
    return np.tile((X*X).sum(-1),[N,1]).T + np.tile((Y*Y).sum(-1),[M,1]) - 2*np.inner(X,Y)
def DataTabulator(X):
    
    Xshape = np.shape(X)
        
    return np.reshape(X,(Xshape[0],np.prod(Xshape[1:])))

def TimeSeriesReshaper(Xflat, numfeatures, subsample = 1, differences = True):
    flatXshape = np.shape(Xflat)
    Xshape = (flatXshape[0],numfeatures,flatXshape[1]/numfeatures)        
    X = np.reshape(Xflat,Xshape)[:,:,::subsample]
    
    if differences:
        return np.diff(X)
    else:    
        return X

def kGA(x,y,scale): 
    return np.exp(-(sqdist(x,y)/(2*(scale**2))+np.log(2-np.exp(-sqdist(x,y)/(2*(scale**2))))))
    
def anova(X,Y,scale,xint,yint):
    K=kGA(X,Y,scale)[:int(xint),:int(yint)]
    return sum(K.diagonal().cumprod())

##Compute cross-matrix
def AnovaKernelXY(X,Y,scale,xint,yint):

    N = np.shape(X)[0]   
    M = np.shape(Y)[0]   
    
    KSeq = np.zeros((N,M))
        
    for row1ind in range(N):
        for row2ind in range(M):
            KSeq[row1ind,row2ind]= anova(X[row1ind].T,Y[row2ind].T,scale,xint[row1ind],yint[row2ind])
          
    
    return KSeq

from sklearn.base import BaseEstimator, TransformerMixin
#Class Bagkernelizer
class AnovaKernelizer(BaseEstimator, TransformerMixin):
    def __init__(self, X = np.zeros((1,2)), scale=1,V=True,
                 numfeatures = 2, subsample = 1, differences =True
                ):
        self.scale=scale
        self.subsample= subsample
        self.numfeatures=numfeatures
        self.differences=differences
        self.V=V
        self.X = X
        
    def fit(self, X, y=None):
        if self.V:
            t=TimeSeriesReshaper(X[:,:-1],self.numfeatures,self.subsample,self.differences)
            v=(X[:,-1]-X[:,-1]%self.subsample)/self.subsample
            self.X=(t,v)
        else:
            t=TimeSeriesReshaper(X,self.numfeatures,self.subsample,self.differences)
            v=t.shape[2]*np.ones(t.shape[0])
            self.X=(t,v)
        return self
        
    def transform(self,Y):
        if self.V:
            y=TimeSeriesReshaper(Y[:,:-1],self.numfeatures,self.subsample,self.differences)
            yt=(Y[:,-1]-Y[:,-1]%self.subsample)/self.subsample
            Y=(y,yt)
        else:
            y= TimeSeriesReshaper(Y,self.numfeatures,self.subsample,self.differences)
            yt=y.shape[2]*np.ones(y.shape[0])
            Y=(y,yt)
        KSeq = AnovaKernelXY(Y[0],self.X[0],self.scale,Y[1]-1,self.X[1]-1)
        return KSeq

from sklearn import svm
from sklearn.pipeline import Pipeline
##pipeline:TGA kernel with SVC
AnovaSVCpipeline=Pipeline([

    ('AnovaKernelizer', AnovaKernelizer()),
    
    ('svc', svm.SVC(kernel = 'precomputed'))

])





















