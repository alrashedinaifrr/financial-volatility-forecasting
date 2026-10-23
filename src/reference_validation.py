"""Independent fixed-parameter GARCH recursion check; not an external ARCH-library fit."""
import numpy as np

def independent_filter(returns_pct, omega, alpha, beta):
    seq=np.asarray(returns_pct,dtype=float)
    states=np.zeros(len(seq))
    states[0]=np.var(seq)
    for t in range(1,len(seq)):
        states[t]=omega+alpha*seq[t-1]**2+beta*states[t-1]
    return states
