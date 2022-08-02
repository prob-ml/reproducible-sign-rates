import numpy as np

def hoeffding_xi(a,axis=None):
    '''
    Computes sum(a)**2 / sum(a**2) under the the convention that
    xi = np.inf when (a==0).all()
    '''

    a=np.require(a).copy()
    if (a<0).any():
        raise ValueError("'a' values should be nonnegative")

    num=np.sum(a,axis=axis)**2
    denom=np.sum(a**2,axis=axis)

    result=np.zeros(num.shape,dtype=float)
    result[num>0]=num[num>0]/denom[num>0]
    result[num==0]=np.inf

    return result
