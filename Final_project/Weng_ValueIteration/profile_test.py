import profile
import operator
import numpy as np
ftype = np.float32


class memoize:
    # from http://avinashv.net/2008/04/python-decorators-syntactic-sugar/
    def __init__(self, function):
        self.function = function
        self.memoized = {}

    def __call__(self, *args):
        try:
            return self.memoized[args]
        except KeyError:
            self.memoized[args] = self.function(*args)
            return self.memoized[args]


@memoize
def fib(n):
    # from http://en.literateprograms.org/Fibonacci_numbers_(Python)
    if n == 0:
        return 0
    elif n == 1:
        return 1
    else:
        return fib(n-1) + fib(n-2)

def fib_seq(n):
    seq = [ ]
    if n > 0:
        seq.extend(fib_seq(n-1))
    seq.append(fib(n))
    return seq


def clean_Points(_points):

        print '_points before'
        print _points

        _points_zero_rows = []
        rows = _points.shape[0]


        for r in range(rows):
            if np.all(_points[r,:] == 0):
                _points_zero_rows.append(r)

        counter = 0
        for r in _points_zero_rows:
            _points = np.delete(_points, (r-counter), axis=0 )
            counter+=1

        print '_points after'
        print _points

        return _points



p = np.array([[  1.60535183e-05 , -2.27292385e-05],
 [  3.34223136e-02 , -3.59819755e-02],
 [ 0.00000000e+00  , 0.00000000e+00],
 [  1.54177360e-05 , -2.17358283e-05],
 [ -5.73247299e-02 ,  1.12755699e-02],
 [  0.00000000e+00  , 0.00000000e+00]])


clean_Points(p)

