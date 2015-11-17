import numpy as np
ftype = np.float32

def pareto_comparison( a, b):
        a = np.array(a, dtype= ftype)
        b = np.array(b, dtype= ftype)

        assert len(a)==len(b), \
                "two vectors don't have the same size"

        return all(a>b)

pareto_comparison([ 0. , 1.]  [ 0.  ,1.])
