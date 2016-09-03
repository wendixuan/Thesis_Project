
######################
# 
#This code was modified basedon the code of 'Kernels for sequentially ordered data' written by F. J Kiraly and H. Oberhauser.
#
#############################################################
######################
#
#This code can be use to do SVC with bag of features kernel, sequential kernel and a simple version of s-sequential kernel 
#
######################

#Addition:
#(1) small_rev():modified funtion of cumsum_rev(), which only consider the diagonal of matrix
#(2) SquizeKernel_D(): the funtion for computing s-sequential kernel values between path 1 and path 2.
#(3) BagSeqKernel(): the funtion for computing bag of feature kernel values between path 1 and path 2.
#(4) BagKernelizer(): a class for computing kernel matrix with the bag of feature
#(5) BagSVCpipeline():pipeline for Bag of feature kernel with SVC
#(6) BagKernelXY() is a function for computing the values of bag of features kernel for two sample pf time series in consistent or inconsistent lengths.

#Modifications are in:
#(1)SeqKernelXY() is modified to be able to work on time series in inconsistent lengths 
#   and be able to compute s-sequential kernel as well (when Dia=True).
#(2)SeqKernelizer() is modified to be able to compute sequential kernel (Dia=False ) and s-sequential kernel(Dia=True) and be able to work with time series in inconsistent lengths (V=True)
#   Notice that when V=True, the dataset should include the vector of original length in the last column of the 2D array.

#####################

import numpy as np
from scipy.sparse.linalg import svds

# In[]
# sqdist

def sqdist(X,Y):
    M = np.shape(X)[0]
    N = np.shape(Y)[0]
    return np.tile((X*X).sum(-1),[N,1]).T + np.tile((Y*Y).sum(-1),[M,1]) - 2*np.inner(X,Y)



# In[]
# cumsum varia

def cumsum_rev_first(array):
    out = np.zeros_like(array)
    out[:-1] = np.cumsum(array[:0:-1],0)[::-1]
    return out

def cumsum_rev(array):
    out = np.zeros_like(array)
    out[:-1, :-1] = np.cumsum(np.cumsum(array[:0:-1, :0:-1], 0), 1)[::-1, ::-1]
    return out


#Modified funtion of cumsum_rev(), which only consider the diagonal of matrix
def small_rev(array):
    out = np.zeros_like(array)
    a=array[:0:-1, :0:-1]
    if a.size>0:
        np.fill_diagonal(a,a.diagonal().cumsum())
    out[:-1, :-1] = a[::-1, ::-1]
    return out

    
def cumsum_mult(array, dims):
    for dimind in dims:
        array = np.cumsum(array, axis = dimind)
    return array
    
def roll_mult(array, shift, dims):
    for dimind in dims:
        array = np.roll(array, shift, axis = dimind)
    return array
    
def makeinds(indlist):
    return np.ix_(*indlist)
    
def cumsum_shift_mult(array, dims):
    array = cumsum_mult(array,dims)
    array = roll_mult(array,1,dims)
    
    arrayshape = array.shape
    indarr = []
    for ind in range(len(arrayshape)):
        indarr = indarr + [range(arrayshape[ind])]
    
    for dimind in dims:
        slicearr = indarr[:]
        slicearr[dimind] = [0]
        array[makeinds(slicearr)] = 0
        
    return array



def rankreduce(array,rankbound):
    arraysvd = svds(array.astype('f'), k = rankbound)
    return np.dot(arraysvd[0],np.diag(arraysvd[1]))        

def rankreduce_batch(arrays,rankbound):
    resultarrays = np.zeros([arrays.shape[0],arrays.shape[1],rankbound])
    for i in range(arrays.shape[0]):
        resultarrays[i,:,:] = rankreduce(arrays[i,:,:], rankbound)
    return resultarrays


# In[]
# x ((obs1,dim)) and y ((obs2,dim)) numpy arrays
#return ((obs1,obs2)) array with kernel as entries

kPolynom = lambda x,y,scale,deg : (1+scale*np.inner(x,y))**deg
kGauss = lambda x,y,scale: np.exp(-(scale**2)*sqdist(x,y)/2)
kEuclid = lambda x,y,scale: scale*np.inner(x,y)
kLaplace = lambda x,y,scale: np.exp(-scale*np.sqrt(np.inner(x-y,x-y)))
kTanH = lambda x,y,off,scale: np.tanh(off+scale*np.inner(x,y))

# In[]

# FUNCTION mirror
#  mirrors an upper triangular kernel matrix, helper for SqizeKernel
mirror = lambda K: K-np.diag(np.diag(K))+np.transpose(K)

# FUNCTION SquizeKernel
#  computes the sequential kernel from a sequential kernel matrix
#
# Inputs:
#  K             the kernel matrix of increments, i.e.,
#                 K[i,j] is the kernel between the i-th increment of path 1,
#                  and the j-th increment of path 2
#  L             an integer \geq 1, representing the level of truncation
# optional:
#  theta         a positive scaling factor for the levels, i-th level by theta^i
#  normalize     whether the output kernel matrix is normalized
#    defaults: theta = 1.0, normalize = False
#
# Output:
#  a real number, the sequential kernel between path 1 and path 2
#
def SqizeKernel(K, L, theta = 1.0, normalize = False):
    #L-1 runs through loop;
    #returns R_ij=(1+\sum_i2>i,j2>j A_i2,j2(1+\sum A_iLjL)...)
    if normalize:
        normfac = np.prod(K.shape)
        I = np.ones(K.shape)
        R = np.ones(K.shape)
        for l in range(L-1):
            R = (I + theta*cumsum_rev(K*R)/normfac)/(1+theta)
        return (1 + theta*np.sum(K*R)/normfac)/(1+theta)
    else:
        I = np.ones(K.shape)
        R = np.ones(K.shape)
        for l in range(L-1):
            R = I + cumsum_rev(K*R) #A*R is componentwise
        return 1 + np.sum(K*R) #outermost bracket: since i1>=1 and not i1>1 we do it outside of loop

# FUNCTION SquizeKernel_D
#  computes the values of  s-sequential kernel matrix 
#
# Inputs:
#  K             the kernel matrix of increments, i.e.,
#                 K[i,j] is the kernel between the i-th increment of path 1,
#                  and the j-th increment of path 2
#  L             an integer \geq 1, representing the level of truncation
# optional:
#  theta         a positive scaling factor for the levels, i-th level by theta^i
#  normalize     whether the output kernel matrix is normalized
#    defaults: theta = 1.0, normalize = False
#
# Output:
#  a real number, the s-sequential kernel between path 1 and path 2
#
def SqizeKernel_D(K, L, theta = 1.0, normalize = False):
    #L-1 runs through loop;
    #returns R_ij=(1+\sum_i2>i,j2>j A_i2,j2(1+\sum A_iLjL)...)
    if normalize:
        normfac = np.prod(K.shape)
        I = np.ones(K.shape)
        R = np.ones(K.shape)
        for l in range(L-1):
            R = (I + theta*small_rev(K*R)/normfac)/(1+theta)
        return (1 + theta*np.sum(K*R)/normfac)/(1+theta)
    else:
        I = np.ones(K.shape)
        R = np.ones(K.shape)
        for l in range(L-1):
            R = I + small_rev(K*R) #A*R is componentwise
        return 1 + np.sum(K*R) #outermost bracket: since i1>=1 and not i1>1 we do it outside of loop

# FUNCTION BagKernel
#  computes the bag of features kernel from a sequential kernel matrix
#
# Inputs:
#  K             the kernel matrix of increments, i.e.,
#                 K[i,j] is the kernel between the i-th increment of path 1,
#                  and the j-th increment of path 2
#
# Output:
#  a real number, the sequential kernel between path 1 and path 2
#
def BagSeqKernel(K):
    normfac = np.prod(K.shape)#the total number of elements in two paths
    return np.sum(K)/normfac
# In[]

# FUNCTION SquizeKernelHO
#  computes the higher order sequential kernel from a sequential kernel matrix
#
# Inputs:
#  K             the kernel matrix of increments, i.e.,
#                 K[i,j] is the kernel between the i-th increment of path 1,
#                  and the j-th increment of path 2
#  L             an integer \geq 1, representing the level of truncation
#  D             an integer \geq 1, representing the order of approximation
# optional:
#  theta         a positive scaling factor for the levels, i-th level by theta^i
#  normalize     whether the output kernel matrix is normalized
#    defaults: theta = 1.0, normalize = False
#
# Output:
#  a real number, the sequential kernel between path 1 and path 2
#
def SqizeKernelHO(K, L, D = 1, theta = 1.0, normalize = False):
    A = np.zeros(np.concatenate(([L,D,D],K.shape)))
    I = np.ones(K.shape)
    
    for l in range(1,L):
        Dprime = min(D, l)
        A[l,0,0,:,:] = K*(I + cumsum_shift_mult(np.sum(A[l-1,:,:,:,:],(0,1)),(0,1) ) )
        for d1 in range(1,Dprime):
            A[l,d1,0,:,:] = A[l,d1,0,:,:] + (1/d1)*K*cumsum_shift_mult(np.sum(A[l-1,d1-1,:,:,:],0),(1))
            A[l,:,d1,:,:] = A[l,0,d1,:,:] + (1/d1)*K*cumsum_shift_mult(np.sum(A[l-1,:,d1-1,:,:],0),(0))
            
            for d2 in range(1,Dprime):
                A[l,d1,d2,:,:] = A[l,d1,d2,:,:] + (1/(d1*d2))*K*cumsum_shift_mult(np.sum(A[l-1,d1-1,d2-1,:,:],0),(0))
                
    return 1 + np.sum(A[L-1,:,:,:,:])



# In[]

import collections

# low-rank decomposition
#  models matrix A = U x V.T
#  U and V should be *arrays*, not *matrices*
LRdec = collections.namedtuple('LRdec', ['U','V'])


# FUNCTION GetLowRankMatrix
#  produce the matrix from the LRdec object
#
# Inputs:
#  K            a LRdec type object
#
# Output:
#  the matrix K.U x K.V.T modelled by the LRdec object
def GetLowRankMatrix(K):
    return np.inner(K.U, K.V)


# FUNCTION AddLowRank
#  efficient computation of sum of low-rank representations
#   using this and then GetLowRankMatrix is more efficient than an
#   explicit computation if the rank of the final matrix is not full
#
# Inputs:
#  K, R           LRdec type objects to add
#
# Output:
#  LRdec type object for sum of K and R

def AddLowRank(K, R):
    return LRdec(np.concatenate((K.U,R.U), axis=1),np.concatenate((K.V,R.V), axis=1))

def AddLowRankOne(U, P):
    return np.concatenate((U,P), axis=1)


def MultLowRank(K, theta):
    return LRdec(theta*K.U, theta*K.V)


# FUNCTION HadamardLowRank
#  efficient computation of Hadamard product of low-rank representations
#   using this and then GetLowRankMatrix is more efficient than an
#   explicit computation if the rank of the final matrix is not full
#
# Inputs:
#  K, R           LRdec type objects to multiply
#
# Output:
#  LRdec type object for Hadamard product of K and R

def HadamardLowRank(K, R):
    rankK = K.U.shape[1]
    rankR = R.U.shape[1]
    U = (np.tile(K.U,rankR)*np.repeat(R.U,rankK,1))
    V = (np.tile(K.V,rankR)*np.repeat(R.V,rankK,1))
    return LRdec(U,V)
    
# multiplies U with every component (1st index) of P
#def HadamardLowRankBatch(U, P):
#    rankU = U.shape[1]
#    N = P.shape[0]
#    rankP = P.shape[2]
#    return (np.repeat(np.repeat(np.array(U,ndmin = 3), rankP, 2),N,0)*np.repeat(P,rankU,2))  

# multiplies U and P component-wise (1st)
def HadamardLowRankBatch(U, P):
    rankU = U.shape[2]
    rankP = P.shape[2]
    return (np.tile(U,rankP)*np.repeat(P,rankU,2))  

# with Nystroem type subsampling
def HadamardLowRankSubS(U, P, rho):
    rankU = U.shape[2]
    rankP = P.shape[2]
    permut = np.sort(np.random.permutation(range(rankU*rankP))[range(rho)])
    return (np.tile(U,rankP)*np.repeat(P,rankU,2))[:,:,permut]
 
    
    
# FUNCTION cumsum_LowRank
# cumsum for LRdec type collections
#  equivalent of cumsum_rev for LRdec type objects
#
# Inputs:
#  K            LRdec type object to cumsum
#
# Output:
#  LRdec type object for cumsum_rev of K

def cumsum_LowRank(K):
    return LRdec(cumsum_rev_first(K.U),cumsum_rev_first(K.V))
    
    
# FUNCTION sum_LowRank
# sum for LRdec type collections
#  equivalent of sum_rev for LRdec type objects
#
# Inputs:
#  K            LRdec type object to sum
#
# Output:
#  LRdec type object for sum of K
def sum_LowRank(K):
    return np.inner(sum(K.U),sum(K.V))
    

# FUNCTION SquizeKernelLowRank
#  computes the sequential kernel from a sequential kernel matrix
#   faster by using a low-rank approximation
#
# Inputs:
#  K              LRdec type object, models low-rank factors
#                   of the increment kernel matrix K such that K = K.U x K.V.T
#                 where K[i,j] is the kernel between the i-th increment of path 1,
#                  and the j-th increment of path 2
#  L             an integer \geq 1, representing the level of truncation
# optional:
#  theta         a positive scaling factor for the levels, i-th level by theta^i
#  normalize     whether the output kernel matrix is normalized
#  rankbound     a hard threshold for the rank of the level matrices
#    defaults: theta = 1.0, normalize = False, rankbound = infinity
#
# Output:
#  a real number, the sequential kernel between path 1 and path 2
#
def SqizeKernelLowRank(K, L, theta = 1.0, normalize = False, rankbound = float("inf")):
    #L-1 runs through loop;
    #returns R_ij=(1+\sum_i2>i,j2>j A_i2,j2(1+\sum A_iLjL)...)
    if normalize:
        K = GetLowRankMatrix(K)
        normfac = np.prod(K.shape)
        I = np.ones(K.shape)
        R = np.ones(K.shape)
        for l in range(L-1):
            R = (I + theta*cumsum_rev(K*R)/normfac)/(1+theta)
        return (1 + theta*np.sum(K*R)/normfac)/(1+theta)
    else:
        I = LRdec(np.ones([K.U.shape[0],1]),np.ones([K.V.shape[0],1]))
         # I = np.ones(K.shape)
        R = I
        for l in range(L-1):
            #todo: execute only if rank is lower than rankbound
            #       reduce to rank
            R = AddLowRank(I,MultLowRank(cumsum_LowRank(HadamardLowRank(K,R)),theta))
            #R=I + cumsum_rev(K*R)
        return 1 + theta*sum_LowRank(HadamardLowRank(K,R)) 
#        return 1 + np.sum(K*R)
        #outermost bracket: since i1>=1 and not i1>1 we do it outside of loop


# FUNCTION SquizeKernelLowRankFast
#  computes the sequential kernel from a sequential kernel matrix
#   faster by using a low-rank approximation
#
# Inputs:
#  K              Array of dimension 3, containing joint low-rank factors
#                  1st index counts sequences
#                  2nd index counts time
#                  3rd index counts features
#                   so K[m,:,:] is the mth factor,
#                    and K[m,:,:] x K[m,:,:]^t is the kernel matrix of the mth factor
#  L             an integer \geq 1, representing the level of truncation
# optional:
#  theta         a positive scaling factor for the levels, i-th level by theta^i
#  normalize     whether the output kernel matrix is normalized
#  rankbound     a hard threshold for the rank of the level matrices
#    defaults: theta = 1.0, normalize = False, rankbound = infinity
#
# Output:
#  a matrix R such that R*R^t is the sequential kernel matrix
#
def SqizeKernelLowRankFast(K, L, theta = 1.0, normalize = False, rankbound = float("inf")):

    if normalize:

        Ksize = K.shape[0]
        B = np.ones([Ksize,1,1])
        R = np.ones([Ksize,1])

        for l in range(L):
            
            P = np.sqrt(theta)*HadamardLowRankBatch(K,B)/Ksize
            B = cumsum_shift_mult(P,[1])
            
            if rankbound < B.shape[2]:
                #B = rankreduce_batch(B,rankbound)
                permut = np.sort(np.random.permutation(range(B.shape[2]))[range(rankbound)])
                B = B[:,:,permut]
                
            R = np.concatenate((R,np.sum(B,axis = 1)), axis=1)/(np.sqrt(1+theta))
            
        return R
        
    else:

        Ksize = K.shape[0]
        B = np.ones([Ksize,1,1])
        R = np.ones([Ksize,1])

        for l in range(L):
            #todo: execute only if rank is lower than rankbound
            #       reduce to rank
            P = np.sqrt(theta)*HadamardLowRankBatch(K,B)
            B = cumsum_shift_mult(P,[1])

            if rankbound < B.shape[2]:
                #B = rankreduce_batch(B,rankbound)
                permut = np.sort(np.random.permutation(range(B.shape[2]))[range(rankbound)])
                B = B[:,:,permut]
                
            R = np.concatenate((R,np.sum(B,axis = 1)), axis=1)

        return R


# In[]
# FUNCTION SeqKernel
#  computes the sequential kernel matrix for a dataset of time series
def SeqKernel(X,kernelfun,L=2,D=1,theta=1.0,normalize = False,lowrank = False,rankbound = float("inf")):
    
    N = np.shape(X)[0]   
    
    KSeq = np.zeros((N,N))
    
    if not(lowrank):
        if D == 1:
            for row1ind in range(N):
                for row2ind in range(row1ind+1):
                    K=kernelfun(X[row1ind].T,Y[row2ind].T)
                    KSeq[row1ind,row2ind] = SqizeKernel(K,L,theta,normalize)
        else:
            for row1ind in range(N):
                for row2ind in range(row1ind+1):
                    KSeq[row1ind,row2ind] = SqizeKernelHO(kernelfun(X[row1ind].T,X[row2ind].T),L,D,theta,normalize)
    else:                

        R = SqizeKernelLowRankFast(X.transpose([0,2,1]), L, theta, normalize)
        KSeq = np.inner(R,R)             
                # todo: kernelfun gives back a LRdec object
                #  for now, linear low-rank approximation is done
                # KSeq[row1ind,row2ind] = SqizeKernelLowRank(kernelfun(X[row1ind].T,X[row2ind].T),L,theta,normalize = True)
        
    return mirror(KSeq) 

def BagKernel(X,kernelfun,xint=[]):
    
    N = np.shape(X)[0]#the number of sample  
    
    KSeq = np.zeros((N,N))#Initial kernel matrix
    
#Using Bag of feature kernel 
    for row1ind in range(N):
        for row2ind in range(row1ind+1):
            K=kernelfun(X[row1ind].T,Y[row2ind].T)
            KSeq[row1ind,row2ind] = BagSeqKernel(K[:xint[row1ind],:xint[row1ind]])    
    return mirror(KSeq) 
    
# FUNCTION SeqKernel
#  computes sequential cross-kernel matrices
def SeqKernelXY(X,Y,kernelfun,L=2,D=1,theta=1.0,normalize = False,lowrank = False,rankbound = float("inf"),xint=[],yint=[],Dia=False):

    N = np.shape(X)[0]   
    M = np.shape(Y)[0]   
    
    KSeq = np.zeros((N,M))
    
    if not(lowrank):
        if D == 1:
            if Dia:
                
                for row1ind in range(N):
                    for row2ind in range(row1ind+1):
                        K=kernelfun(X[row1ind].T,Y[row2ind].T)
                        KSeq[row1ind,row2ind] = SqizeKernel(K[:xint[row1ind],:yint[row2ind]],L,theta,normalize)
            else:
                for row1ind in range(N):
                    for row2ind in range(M):
                        K=kernelfun(X[row1ind].T,Y[row2ind].T)
                        KSeq[row1ind,row2ind] = SqizeKernel(K[:xint[row1ind],:yint[row2ind]],L,theta,normalize)
       
        else:
            for row1ind in range(N):
                for row2ind in range(M):
                    KSeq[row1ind,row2ind] = SqizeKernelHO(kernelfun(X[row1ind].T,Y[row2ind].T),L,D,theta,normalize)
    else:
        
        KSeq = np.inner(SqizeKernelLowRankFast(X.transpose([0,2,1]), L, theta, normalize, rankbound),SqizeKernelLowRankFast(Y.transpose([0,2,1]), L, theta, normalize, rankbound))             
        #KSeq = np.inner(SqizeKernelLowRankFast(X, L, theta, normalize),SqizeKernelLowRankFast(Y, L, theta, normalize))             
                
    return KSeq
# FUNCTION BagKernelXY
#  computes sequential cross-kernel matrices    
def BagKernelXY(X,Y,kernelfun,xint,yint):

    N = np.shape(X)[0]   
    M = np.shape(Y)[0]   
    
    KSeq = np.zeros((N,M))
        
    for row1ind in range(N):
        for row2ind in range(M):
            K=kernelfun(X[row1ind].T,Y[row2ind].T)
            KSeq[row1ind,row2ind] = BagSeqKernel(K[:xint[row1ind],:yint[row2ind]])
    
    return KSeq
# FUNCTION SeqKernelXY_D
#  computes sequential cross-kernel matrices    

# In[]
# FUNCTION DataTabulator(X)
def DataTabulator(X):
    
    Xshape = np.shape(X)
        
    return np.reshape(X,(Xshape[0],np.prod(Xshape[1:])))




# In[]
# FUNCTION TimeSeriesReshaper
#  makes a 3D time series array out of a 2D data array
def TimeSeriesReshaper(Xflat, numfeatures, subsample = 1, differences = True):
    flatXshape = np.shape(Xflat)
    Xshape = (flatXshape[0], numfeatures, flatXshape[1]/numfeatures)        
    X = np.reshape(Xflat,Xshape)[:,:,::subsample]
    
    if differences:
        return np.diff(X)
    else:    
        return X
        

# In[3]

# CLASS SeqKernelizer
#  pipelines pre-processing of a time series datset with support vector classifier
#
# parameters:
#  Level, theta: parameters in of the sequentialization
#   Level = cut-off degree
#   theta = scaling factor
#  kernel, scale, deg: parameter for the primary kernel
#   kernel = name of the kernel used: linear, Gauss, Laplace, poly
#   scale = scaling constant, multiplicative to scalar product
#   deg = degree, for polynomial kernel
#  subsample, numfeatures, differences:
#   pre-processing parameters for time series.
#    numfeatures = number of features per time point, for internal reshaping
#    subsample = time series is subsampled to every subsample-th time point
#    differences = whether first differences are taken or not
#    lowrank = whether low-rank approximations are used or not
# New addition:
##   V: True=time series in inconsistent length
#    Dia: True=a simple version of s-sequential kernel False=Sequential kernel 
#
from sklearn.base import BaseEstimator, TransformerMixin

class SeqKernelizer(BaseEstimator, TransformerMixin):
    def __init__(self, Level = 2, Degree = 1, theta = 1, kernel = 'linear', 
                 scale = 1, deg = 2, X = np.zeros((1,2)), V=False,
                 numfeatures = 2, subsample = 100, differences = True, Dia = False,
                 normalize = False, lowrank = False, rankbound = float("inf")):
        self.Level = Level
        self.Degree = Degree
        self.theta = theta
        self.subsample = subsample
        self.kernel = kernel
        self.scale = scale
        self.deg = deg
        self.numfeatures = numfeatures
        self.differences = differences
        self.normalize = normalize
        self.lowrank = lowrank
        self.rankbound = rankbound
        self.X = X
        self.V=V
        self.Dia=Dia
        
    def fit(self, X, y=None):
        if self.V:
            t=TimeSeriesReshaper(X[:,:-1],self.numfeatures,self.subsample,self.differences)
            v=(X[:,-1]-X[:,-1]%self.subsample)/self.subsample
            self.X=(t,v)
        else:
            t=TimeSeriesReshaper(X,self.numfeatures,self.subsample,self.differences)
            n=int(t.shape[2]/self.subsample)
            v=n*np.ones(t.shape[0])
            self.X=(t,v)
        return self
        
    def transform(self, Y):
        
        if self.V:
            y=TimeSeriesReshaper(Y[:,:-1],self.numfeatures,self.subsample,self.differences)
            yt=(Y[:,-1]-Y[:,-1]%self.subsample)/self.subsample
            Y=(y,yt)
        else:
            y= TimeSeriesReshaper(Y,self.numfeatures,self.subsample,self.differences)
            n=int(y.shape[2]/self.subsample)
            yt=n*np.ones(y.shape[0])
            
            Y=(y,yt)
        
        kPolynom = lambda x,y,scale,deg : (1+scale*np.inner(x,y))**deg
        kGauss = lambda x,y,scale: np.exp(-(scale**2)*sqdist(x,y)/2)
        kGA= lambda x,y,scale: np.exp(-(sqdist(x,y)/(2*(scale**2))+np.log(2-np.exp(-sqdist(x,y)/(2*(scale**2))))))
        kEuclid = lambda x,y,scale: scale*np.inner(x,y)
        kLaplace = lambda x,y,scale: np.exp(-scale*np.sqrt(np.inner(x-y,x-y)))
        
        def kernselect(kername):
            switcher = {
                'linear': lambda x,y: kEuclid(x,y,self.scale),
                'Gauss': lambda x,y: kGauss(x,y,self.scale),
                'GA': lambda x,y: kGA(x,y,self.scale),
                'Laplace': lambda x,y: kLaplace(x,y,self.scale),
                'poly': lambda x,y: kPolynom(x,y,self.scale,self.deg),
                }
            return switcher.get(kername, "nothing")
            
        KSeq = SeqKernelXY(Y[0],self.X[0],kernselect(self.kernel),self.Level,self.Degree,self.theta,self.normalize,self.lowrank,self.rankbound,Y[1]-1,self.X[1]-1,self.Dia)
        
        return KSeq



# In[]
# CLASS TimeSeriesPreprocesser
#  for pre-processing of time series type features
#
# parameters:
#  numfeatures = number of features per time point, for internal reshaping
#  subsample = time series is subsampled to every subsample-th time point
#  differences = whether first differences are taken or not
#
class TimeSeriesPreprocesser(BaseEstimator, TransformerMixin):
    def __init__(self, numfeatures = 2, subsample = 100, scale = 1, differences = True):
        self.subsample = subsample
        self.numfeatures = numfeatures
        self.scale = scale
        self.differences = differences
     
        
    def fit(self, X, y=None):
        return self
        
    def transform(self, Y):
        
        Y = self.scale*TimeSeriesReshaper(Y,self.numfeatures,self.subsample,self.differences)
        
        return DataTabulator(Y)

#Class Bagkernelizer

# parameters:

#   kernel = name of the kernel used: linear, Gauss, Laplace, poly
#   scale = scaling constant, multiplicative to scalar product
#   deg = degree, for polynomial kernel
#    numfeatures = number of features per time point, for internal reshaping
#    subsample = time series is subsampled to every subsample-th time point
#    differences = whether first differences are taken or not
#   V: True=time series in inconsistent length

class BagKernelizer(BaseEstimator, TransformerMixin):
    def __init__(self,kernel = 'linear', 
                 scale = 1, deg = 2, X=0,V=False,
                 numfeatures = 2, subsample = 100, differences = True 
                ):

        self.subsample = subsample
        self.kernel = kernel
        self.scale = scale
        self.deg = deg
        self.numfeatures = numfeatures
        self.differences = differences
        self.X = X
        self.V=V
        
    def fit(self, X, y=None):
        if self.V:
            t=TimeSeriesReshaper(X[:,:-1],self.numfeatures,self.subsample,self.differences)
            v=(X[:,-1]-X[:,-1]%self.subsample)/self.subsample
            self.X=(t,v)
        else:
            t=TimeSeriesReshaper(X,self.numfeatures,self.subsample,self.differences)
            n=int(t.shape[2]/self.subsample)
            v=n*np.ones(t.shape[0])
            self.X=(t,v)
        return self
        
    
        
    def transform(self,Y):
        if self.V:
            y=TimeSeriesReshaper(Y[:,:-1],self.numfeatures,self.subsample,self.differences)
            yt=(Y[:,-1]-Y[:,-1]%self.subsample)/self.subsample
            Y=(y,yt)
        else:
            y= TimeSeriesReshaper(Y,self.numfeatures,self.subsample,self.differences)
            n=int(y.shape[2]/self.subsample)
            yt=n*np.ones(y.shape[0])
            Y=(y,yt)
        
        kPolynom = lambda x,y,scale,deg : (1+scale*np.inner(x,y))**deg
        kGauss = lambda x,y,scale: np.exp(-(scale**2)*sqdist(x,y)/2)
        kGA= lambda x,y,scale: np.exp(-(sqdist(x,y)/(2*(scale**2))+np.log(2-np.exp(-sqdist(x,y)/(2*(scale**2))))))
        kEuclid = lambda x,y,scale: scale*np.inner(x,y)
        kLaplace = lambda x,y,scale: np.exp(-scale*np.sqrt(np.inner(x-y,x-y)))
        
        def kernselect(kername):
            switcher = {
                'linear': lambda x,y: kEuclid(x,y,self.scale),
                'Gauss': lambda x,y: kGauss(x,y,self.scale),
                'GA': lambda x,y: kGA(x,y,self.scale),
                'Laplace': lambda x,y: kLaplace(x,y,self.scale),
                'poly': lambda x,y: kPolynom(x,y,self.scale,self.deg),
                }
            return switcher.get(kername, "nothing")
            
        KSeq = BagKernelXY(Y[0],self.X[0],kernselect(self.kernel),Y[1]-1,self.X[1]-1)

        return KSeq


# In[]

from sklearn import svm
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# In[]

# pipeline: sequential kernel with SVC
SeqSVCpipeline = Pipeline([

    ('SeqKernelizer', SeqKernelizer()),
    
    ('svc', svm.SVC(kernel = 'precomputed'))

])


#pipeline:Bag of feature kernel with SVC
BagSVCpipeline=Pipeline([

    ('BagKernelizer', BagKernelizer()),
    
    ('svc', svm.SVC(kernel = 'precomputed'))

])


# pipeline: pre-processing and SVC, no sequential kernel - baseline
TimeSVCpipeline = Pipeline([

    ('TimeSeriesPP', TimeSeriesPreprocesser()),
    
    ('svc', svm.SVC())

])


# pipeline: pre-processing, standardization, and SVC, no sequential kernel - baseline
TimeStdSVCpipeline = Pipeline([

    ('TimeSeriesPP', TimeSeriesPreprocesser()),

    ('standardize', StandardScaler()),
    
    ('svc', svm.SVC())

])
            
