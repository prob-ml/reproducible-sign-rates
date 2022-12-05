import dataclasses

import tensorflow as tf
import numpy as np

import scipy as sp

@dataclasses.dataclass
class AUROCResults:
    '''
    Assume we have K pairs of distributions, `(p[0],q[0]),....(p[K],q[K])`.
    This class stores information about estimating AUROC(p[i],q[i]) for each i.

    Attributes:
        n_X: number of iid samples from each pi
        n_Y: number of iid samples from each qi
        aurocs: empirical AUROCs
        tie_terms: tie correction terms for calculating variance of empirical AUROCs


    '''
    n_X: np.ndarray
    n_Y: np.ndarray
    aurocs: np.ndarray
    tie_terms: np.ndarray

    def __len__(self):
        return len(self.aurocs)

    def mann_whitney_zscore(self,continuity_correction=True,tie_correction=True):
        '''
        Deviations of the empirical AUROCs from 0.5, divided by their estimated
        variance under the null hypothesis that the true AUROCs are exactly 0.5.  In the limit
        as n_X and n_Y go to infinity under this null, these objects will asymptotically
        converge to standard normals (i.e. z-scores).  This approximation is often used to compute
        p-values for Mann Whitney tests.

        Two remarks.

        * See `mann_whitney_variance` for additional details on the estimated variance and how it is affected by "tie_correction".
        * When continuity_correction=True, a small fudge factor is added to ensure that a hypothesis test based on these deviations is more conservative.
        '''
        delta=self.aurocs-.5

        n_product=self.n_X*self.n_Y

        if continuity_correction:
            # U --> U - .5
            # auroc --> auroc - .5/(n1*n2)
            mod=.5/np.clip(n_product,1,np.inf)

            morethanmod=delta>=mod
            delta[morethanmod]=delta[morethanmod]-mod[morethanmod]

            lessthanmod=delta<=-mod
            delta[lessthanmod]=delta[lessthanmod]+mod[lessthanmod]

        # get variance
        std=np.sqrt(self.mann_whitney_variance(tie_correction=tie_correction))

        # handle the case where its nothing but ties
        trivial=np.abs(delta)==0
        assert (std[~trivial]>0).all()
        std[trivial]=1

        # done!
        return delta/std

    def mann_whitney_variance(self,tie_correction=True):
        '''
        Variance of the empirical AUROCs under the null hypothesis that the AUROC is .5.

        If tie_correction is used, exact variances are computed.   This will
        lead to much smaller variances when there are lots of ties (i.e., larger zscores), but it will also
        tend to emphasize discoveries with low effect sizes.

        If tie_correction is not used, we will obtain a larger variance (i.e., *conservative* z-scores) leading to
        larger p-values; this variance can be understood as the variance we would obtain if infinitesimal
        noise was added to each sample to break all ties.
        '''
        if tie_correction and (self.tie_terms is None):
            raise ValueError("Tie terms were not computed; cannot compute variances with tie correction")

        nmins=np.min([self.n_X,self.n_Y],axis=0)

        variances=np.zeros(len(self))

        ############
        # when there's no data, nobody
        # should be using the variance term!
        variances[nmins==0] = np.nan

        #########
        # when there is, we have the mann-whitney variance
        n1=self.n_X[nmins>0]
        n2=self.n_Y[nmins>0]
        n=n1+n2
        mw_variance = n+1
        if tie_correction:
            tie_terms=self.tie_terms[nmins>0]
            corrections = (tie_terms / (n*(n-1)))
            variances[nmins>0] = (mw_variance - corrections)/(12*n1*n2)
        else:
            variances[nmins>0] = mw_variance/(12*n1*n2)

        return variances


    def conservative_variance(self):
        '''
        Computes a conservative estimate of the
        variance of the empirical AUROCs as estimators
        of the true probabilistic index, using the upper
        bound

        Var(auroc) <= .25/min(n_X,n_Y)

        This upper bound is applicable even if the null hypothesis does not hold.
        '''
        nmins=np.min([self.n_X,self.n_Y],axis=0)

        variances=np.zeros(len(self))

        variances[nmins==0] = 1
        variances[nmins>0] = .25/nmins[nmins>0]

        return variances


def calc_midsigns(x,y):
    '''
    Input:
    * x -- batch x n-vector
    * y -- batch x m-vector
    * [optional] presorted -- boolean

    Output:
    * midranks -- batch x n-vector

    with

      midsigns[b,i] = np.sum(np.sign(x[b,i]-y[b]))

    If presorted, we assume y is sorted.  If y is not sorted
    but presorted = True the behavior is undefined.
    '''

    y=tf.sort(y,axis=-1)

    m=tf.shape(y)[-1]

    L=tf.searchsorted(y,x,side='right')
    R=tf.searchsorted(y,x,side='left')

    return L +R-m

def calc_ties(X):
    '''
    Input:
    * x -- batch x n-vector

    Output is tie_term -- batch

    Tie correction term for batch b is defined by

    vals,counts=unique(X[b],return_counts=True)
    tie_term = sum(counts**3-counts)

    '''

    shp=tf.shape(X)
    batch=shp[0]
    n=shp[1]

    # get segments
    X=tf.sort(X,axis=-1)
    X=X[:,1:]-X[:,:-1]
    X=tf.cast(X!=0,dtype=tf.int32)
    X=tf.concat([tf.zeros((batch,1),dtype=tf.int32),X],axis=-1)
    X=tf.cumsum(X,axis=-1)

    # count segment sizes
    XC=tf.range(batch)[:,None]*tf.ones(n,dtype=X.dtype)[None,:]
    coords=tf.stack([
        tf.reshape(XC,(batch*n,)),
        tf.reshape(X,(batch*n,))
    ],axis=1)
    ones=tf.ones(batch*n,dtype=tf.int32)
    sizes=tf.scatter_nd(coords,ones,shp)

    # done!
    return tf.reduce_sum(sizes**3-sizes,axis=-1)

@tf.function(input_signature=[
    tf.TensorSpec(shape=(None,None)),
    tf.TensorSpec(shape=(None,None)),
])
def _calc_auroc_regular_batched(X,Y):
    '''
    Input:
    * x -- batch x n-vector  (the "control")
    * y -- batch x m-vector  (the "treatment")

    Output
    * aurocs
    * tie_terms
    '''

    midsigns=tf.cast(calc_midsigns(Y,X),dtype=tf.float32)
    n = tf.cast(tf.shape(X)[-1],dtype=tf.float32)
    aurocs=.5+.5*tf.reduce_mean(midsigns,axis=-1)/n

    return aurocs

@tf.function(input_signature=[
    tf.TensorSpec(shape=(None,None)),
    tf.TensorSpec(shape=(None,None)),
])
def _calc_ties_regular_batched(X,Y):
    '''
    Input:
    * x -- batch x n-vector  (the "control")
    * y -- batch x m-vector  (the "treatment")

    Output is tie_terms
    '''

    XY=tf.concat([X,Y],axis=-1)
    tie_terms=calc_ties(XY)

    return tie_terms

def calc_aurocs_from_matrices(X,Y,calc_ties=True):
    '''
    Assume K pairs of distributions, `(p[0],q[0]),....(p[K],q[K])`.
    For pair `k` further assume we have obtained samples of the following form.

    ```
    X[k,0],X[k,1],...,X[k,n] ~ iid p[k]
    Y[k,0],Y[k,1],...,X[k,m] ~ iid q[k]
    ```

    Note that for each k we have exactly n samples from p[k] and m samples from
    q[k].

    This function returns an AUROCResults object which stores information
    about estimates for AUROC(p[k],q[k]) for each k.

    Args:
        X (b x n, matrix): samples from p
        Y (b x m, matrix): samples from q
        calc_ties (bool): whether to calculate tie correction terms
    '''

    if (X.shape[1]==0) or (Y.shape[1]==0): # no samples!
        return AUROCResults(
            n_X=X.shape[1]*np.ones(X.shape[0]),
            n_Y=Y.shape[1]*np.ones(Y.shape[0]),
            aurocs=np.ones(X.shape[0])*.5,
            tie_terms=np.zeros(X.shape[0]),
        )
    else:

        X=tf.convert_to_tensor(X,dtype=tf.float32)
        Y=tf.convert_to_tensor(Y,dtype=tf.float32)
        aurocs=_calc_auroc_regular_batched(X,Y).numpy().astype(float)
        tie_terms=_calc_ties_regular_batched(X,Y).numpy().astype(int) if calc_ties else None

        return AUROCResults(
            X.shape[1]*np.ones(X.shape[0],dtype=np.int),
            Y.shape[1]*np.ones(Y.shape[0],dtype=np.int),
            aurocs,
            tie_terms,
        )

def calc_aurocs(X,Y,calc_ties=True):
    '''
    Assume K pairs of distributions, `(p[0],q[0]),....(p[K],q[K])`.
    For pair `k` further assume we have obtained samples of the following form.

    ```
    X[k][0],X[k][1],...,X[k][n[k]] ~ iid p[k]
    Y[k][0],Y[k][1],...,X[k][m[k]] ~ iid q[k]
    ```

    This function returns an AUROCResults object which stores information
    about estimates for AUROC(p[k],q[k]) for each k.

    Note: this function is typically about 40 times slower than calc_aurocs_from_matrices,
    in which we may assume that n[0]=n[1]=...n[K-1] and m[0]=m[1]=...m[K-1].

    Args:
        X (K x *, list of vectors): samples from p
        Y (K x *, list of vectors): samples from q
        calc_ties (bool): whether to calculate tie correction terms
    '''

    if len(X)!=len(Y):
        raise ValueError("X and Y should have the same length")

    infos=[calc_aurocs_from_matrices(x[None],y[None],calc_ties) for (x,y) in zip(X,Y)]

    return AUROCResults(
        np.concatenate([p.n_X for p in infos]),
        np.concatenate([p.n_Y for p in infos]),
        np.concatenate([p.aurocs for p in infos]),
        np.concatenate([p.tie_terms for p in infos]) if calc_ties else None,
    )

def _test_aurocs():
    rng=np.random.default_rng()
    X=rng.integers(0,5,size=(2,50))
    Y=rng.integers(0,5,size=(2,30))+1
    ar=calc_aurocs(X,Y)
    pvalue=sp.stats.norm.cdf(-np.abs(ar.mann_whitney_zscore()))*2
    rez=sp.stats.mannwhitneyu(X[0],Y[0],alternative='two-sided',use_continuity=True)
    assert np.allclose(rez.pvalue,pvalue[0]),f"{rez.pvalue} vs {pvalue}"

    X=rng.integers(0,5,size=(2,50))+1
    Y=rng.integers(0,5,size=(2,30))
    ar=calc_aurocs(X,Y)
    pvalue=sp.stats.norm.cdf(-np.abs(ar.mann_whitney_zscore()))*2
    rez=sp.stats.mannwhitneyu(X[0],Y[0],alternative='two-sided',use_continuity=True)
    assert np.allclose(rez.pvalue,pvalue[0]),f"{rez.pvalue} vs {pvalue}"