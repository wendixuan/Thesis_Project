###############################

#############################################################################################################
#
#
#
#   This is codes for the simulationexperiemnents for comparing the running time of sequential kernel and
#   s-sequential kernel
#
#
###########################################################################################################
import time
import numpy as np
################################
#1. Functions for exeriments
################################


#
#Funtion of S-Sequential kernel (considering the diagnale elements )
#Note that this function based on the code of Kiraly(2016)
#
def small_rev(array):
    out = np.zeros_like(array)
    a=array[:0:-1, :0:-1]
    #if a.size>0:
    np.fill_diagonal(a,a.diagonal().cumsum())
    out[:-1, :-1] = a[::-1, ::-1]
    return out

def SqizeKernel_D(K, L):
    #L-1 runs through loop;
    #returns R_ij=(1+\sum_i2>i,j2>j A_i2,j2(1+\sum A_iLjL)...)

    I = np.ones(K.shape)
    R = np.ones(K.shape)
    for l in range(L-1):
        R = I + small_rev(K*R) #A*R is componentwise
    return 1 + np.sum(K*R) #outermost bracket: since i1>=1 and not i1>1 we do it outside of loop

#
#Funtion of Sequential kernel 
#Note that this function from the code of Kiraly(2016)
#
def cumsum_rev(array):
    out = np.zeros_like(array)
    out[:-1, :-1] = np.cumsum(np.cumsum(array[:0:-1, :0:-1], 0), 1)[::-1, ::-1]
    return out
    
#Note that this function from the code of Kiraly(2016)
def SqizeKernel(K, L):
    #L-1 runs through loop;
    #returns R_ij=(1+\sum_i2>i,j2>j A_i2,j2(1+\sum A_iLjL)...)

    I = np.ones(K.shape)
    R = np.ones(K.shape)
    for l in range(L-1):
        R = I + cumsum_rev(K*R) #A*R is componentwise
    return 1 + np.sum(K*R) #outermost bracket: since i1>=1 and not i1>1 we do it outside of loop

################################
#2. simulation experiemnts
#
#  Sequential kernel matrix and s-sequential kernel matrix are calculated for 100 datasets with 30 samples.
#  The average running time of 100 repeats are recorded and printed
#
################################






##time series=50
A=np.zeros((100,4,1))
B=np.zeros((100,4,1))
for N in range(100):
    a=np.zeros((4,1))
    b=np.zeros((4,1))
    for n in range(900):

        K=10*np.random.random([50,50])
        for d in range(2,6):
            s=time.clock()
            SqizeKernel_D(K,d)
            e=time.clock()
            a[d-2]=a[d-2] +e-s

            s1=time.clock()
            SqizeKernel(K,d)
            e1=time.clock()
            b[d-2]=b[d-2]+e1-s1
    A[N,:,:]=a
    B[N,:,:]=b

print ('s-sequential kernel:',np.mean(A,0).T,np.std(A,0).T)
print ('sequential kernel',np.mean(B,0).T,np.std(B,0).T)


##time series=100
A=np.zeros((100,4,1))
B=np.zeros((100,4,1))
for N in range(100):
    a=np.zeros((4,1))
    b=np.zeros((4,1))
    for n in range(900):

        K=10*np.random.random([100,100])
        for d in range(2,6):
            s=time.clock()
            SqizeKernel_D(K,d)
            e=time.clock()
            a[d-2]=a[d-2] +e-s

            s1=time.clock()
            SqizeKernel(K,d)
            e1=time.clock()
            b[d-2]=b[d-2]+e1-s1
    A[N,:,:]=a
    B[N,:,:]=b

print ('s-sequential kernel:',np.mean(A,0).T,np.std(A,0).T)
print ('sequential kernel',np.mean(B,0).T,np.std(B,0).T)



##time series=500
A=np.zeros((100,4,1))
B=np.zeros((100,4,1))
for N in range(100):
    a=np.zeros((4,1))
    b=np.zeros((4,1))
    for n in range(900):

        K=10*np.random.random([500,500])
        for d in range(2,6):
            s=time.clock()
            SqizeKernel_D(K,d)
            e=time.clock()
            a[d-2]=a[d-2] +e-s

            s1=time.clock()
            SqizeKernel(K,d)
            e1=time.clock()
            b[d-2]=b[d-2]+e1-s1
    A[N,:,:]=a
    B[N,:,:]=b

print ('s-sequential kernel:',np.mean(A,0).T,np.std(A,0).T)
print ('sequential kernel',np.mean(B,0).T,np.std(B,0).T)



##time series=1000
##time series=100
A=np.zeros((100,4,1))
B=np.zeros((100,4,1))
for N in range(100):
    a=np.zeros((4,1))
    b=np.zeros((4,1))
    for n in range(900):

        K=10*np.random.random([1000,1000])
        for d in range(2,6):
            s=time.clock()
            SqizeKernel_D(K,d)
            e=time.clock()
            a[d-2]=a[d-2] +e-s

            s1=time.clock()
            SqizeKernel(K,d)
            e1=time.clock()
            b[d-2]=b[d-2]+e1-s1
    A[N,:,:]=a
    B[N,:,:]=b

print ('s-sequential kernel:',np.mean(A,0).T,np.std(A,0).T)
print ('sequential kernel',np.mean(B,0).T,np.std(B,0).T)



























