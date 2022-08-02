import numpy as np

def _sweep_unsigned(rho,Yhat,Y,thresholds):
    srt=np.argsort(rho)
    local_nguesses=np.searchsorted(rho[srt],thresholds)
    local_nagreements=np.cumsum(Yhat[srt]==Y[srt])
    local_nagreements_by_thresh = np.zeros(len(thresholds),dtype=int)
    local_nagreements_by_thresh[local_nguesses>0]=local_nagreements[local_nguesses[local_nguesses>0]-1]
    return local_nguesses,local_nagreements_by_thresh

def sweep(rho,Yhat,Y,thresholds):
    up=Yhat>0
    down=Yhat<0
    r_up,a_up = _sweep_unsigned(rho[up],Yhat[up],Y[up],thresholds)
    r_down,a_down = _sweep_unsigned(rho[down],Yhat[down],Y[down],thresholds)

    return r_up,r_down,a_up,a_down