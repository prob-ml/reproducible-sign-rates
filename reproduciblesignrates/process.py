import dataclasses

import numpy as np
import scipy as sp

import scipy.stats

import reproduciblesignrates.sweeps
import reproduciblesignrates.hoeffding

def _rsp(rejections,agreements):
    '''
    agreements / rejections with the convention that 0/0 = 1
    '''

    typeS_error_rate=(rejections-agreements)/np.clip(rejections,1,np.inf)
    return 1-typeS_error_rate

@dataclasses.dataclass
class ReproducibleSignRateInfo:
    '''
    Stores various properties of two experimental replicates
    in terms of how well they agree.

    Attributes:
        logrho (N vector): confidence in null hypotheses (from training replicate)
        Yhat (N vector): estimate of sign of parameters (from training replicate)
        Yhat (N vector): estimate of sign of parameters (from testing replicate)
        thresholds (L vector): different thresholds on logrho to use
        subexperiment_ids (N vector): subexperiment assocaited with each parameter
        RSP (L vector): reproducible sign proportion at each threshold
        median_rejections (L vector): median rejections per subexperiment at each threshold
    '''

    logrho: np.ndarray
    Yhat: np.ndarray
    Y: np.ndarray
    subexperiment_ids: np.ndarray
    thresholds: np.ndarray

    def confidence_interval(self,alpha,alternative='two-sided'):
        '''
        Computes confidence interval with (1-alpha) nominal coverage probability.
        Returns a lower and upper bound for each threshold in `self.thresholds`.
        '''

        if alternative=='two-sided':
            eps=np.sqrt(-np.log(alpha/2)/(2*self.xi))
            return (
                np.clip(self.RSP-eps,0,1),
                np.clip(self.RSP+eps,0,1)
            )
        elif alternative=='lower-bound':
            eps=np.sqrt(-np.log(alpha)/(2*self.xi))
            return (
                np.clip(self.RSP-eps,0,1),
                np.ones(self.n_thresholds)
            )
        elif alternative=='upper-bound':
            eps=np.sqrt(-np.log(alpha)/(2*self.xi))
            return (
                np.zeros(self.n_thresholds),
                np.clip(self.RSP+eps,0,1),
            )
        else:
            raise NotImplementedError(alternative)


    def __post_init__(self):
        subexperiments,subexperiment_idxs=np.unique(
            self.subexperiment_ids,return_inverse=True)
        self.n_subexperiments=len(subexperiments)
        self.n_thresholds=len(self.thresholds)

        self.rejections_up=np.zeros((self.n_subexperiments,self.n_thresholds))
        self.rejections_down=np.zeros((self.n_subexperiments,self.n_thresholds))
        self.agreements_up=np.zeros((self.n_subexperiments,self.n_thresholds))
        self.agreements_down=np.zeros((self.n_subexperiments,self.n_thresholds))

        for i in range(self.n_subexperiments):
            indicator=subexperiment_idxs==i
            r_up,r_down,a_up,a_down=reproduciblesignrates.sweeps.sweep(
                self.logrho[indicator],self.Yhat[indicator],
                self.Y[indicator],self.thresholds)
            self.rejections_up[i]=r_up
            self.rejections_down[i]=r_down
            self.agreements_up[i]=a_up
            self.agreements_down[i]=a_down


        self.agreements=self.agreements_up+self.agreements_down
        self.rejections=self.rejections_up+self.rejections_down

        self.RSP = _rsp(np.sum(self.rejections,axis=0),np.sum(self.agreements,axis=0))

        self.total_rejections=np.sum(self.rejections,axis=0)
        self.median_rejections=np.median(self.rejections,axis=0)

        self.xi=reproduciblesignrates.hoeffding.hoeffding_xi(self.rejections,axis=0)

def _check_sign_format(Y):
    Y=np.require(Y)

    if Y.dtype.kind!='i':
        raise ValueError("Sign estimate vector should be integer-valued")

    set(np.unique(Y))
    if not set(np.unique(Y)).issubset({-1,0,1}):
        raise ValueError("Sign estimate vector may only contain values {-1,0,1}")

    return Y

def process_from_matrices(logrho,Yhat,Y,n_thresholds=1000):
    '''
    Returns a ReproducibleSignRateInfo object, storing various properties of two
    experimental replicates in terms of how well they agree.  Each row of logrho, Yhat, Y is assumed
    to correspond to a different subexperiment.

    If Yhat[i,j]==0, then parameter associated with index i,j is ignored (regardless of the associated rho value).

    Args:
        logrho (NxM matrix): confidence in a null hypothesis, expressed as a log p-value (from training replicate)
        Yhat (NxM matrix, int): estimate of the sign of a parameter theta (from training replicate)
        Y (NxM matrix, int): another estimate (from testing replicate)
        n_thresholds (int): number of thresholds on logrho to use
    '''
    logrho=np.require(logrho,dtype=float)
    Yhat=_check_sign_format(Yhat)
    Y=_check_sign_format(Y)

    if len(logrho.shape)!=2:
        raise ValueError("rho should be a matrix where each row is a different subexperiment")
    if len(Yhat.shape)!=2:
        raise ValueError("Yhat should be a matrix where each row is a different subexperiment")
    if len(Y.shape)!=2:
        raise ValueError("Y should be a matrix where each row is a different subexperiment")

    if logrho.shape!=Yhat.shape or Yhat.shape!=Y.shape:
        raise ValueError("rho,Yhat,Y should all have the same shape")

    subexperiment_ids=np.outer(
        np.r_[0:logrho.shape[0]],
        np.ones(logrho.shape[1])
    )

    thresholds=np.r_[logrho.min():logrho.max():n_thresholds*1j]

    return ReproducibleSignRateInfo(
        logrho.ravel(),
        Yhat.ravel(),
        Y.ravel(),
        subexperiment_ids.ravel(),
        thresholds)

