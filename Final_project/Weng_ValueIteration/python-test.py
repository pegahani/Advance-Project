import numpy as np
from scipy.spatial.distance import cdist as linfDistance

exact = [  3.80165195,  14.22766876]
vector_list_abvi = [np.array([ 0.,  0.], dtype=np.float32), np.array([ 0.66666669,  0.        ], dtype=np.float32),
                    np.array([ 5.78545332,  5.85214043], dtype=np.float32), np.array([ 5.98728991,  6.06842422], dtype=np.float32),
                    np.array([ 6.17903376,  6.27389479], dtype=np.float32), np.array([ 6.3611908 ,  6.46909142], dtype=np.float32),
                    np.array([ 6.53423977,  6.65452814], dtype=np.float32), np.array([ 6.69863605,  6.83069324], dtype=np.float32),
                    np.array([ 6.8548131 ,  6.99804926], dtype=np.float32), np.array([ 7.00318098,  7.15703869], dtype=np.float32),
                    np.array([ 7.14413071,  7.30807781], dtype=np.float32), np.array([ 7.27803278,  7.45156527], dtype=np.float32),
                    np.array([ 7.40524006,  7.5878787 ], dtype=np.float32), np.array([ 7.52608681,  7.71737528], dtype=np.float32),
                    np.array([ 7.6408906 ,  7.84039831], dtype=np.float32), np.array([ 7.7499547 ,  7.95726967], dtype=np.float32),
                    np.array([ 7.85356522,  8.06829834], dtype=np.float32), np.array([ 7.95199585,  8.17377472], dtype=np.float32),
                    np.array([ 8.04550552,  8.27397728], dtype=np.float32), np.array([ 8.13433838,  8.36916924], dtype=np.float32),
                    np.array([ 8.21873093,  8.45960236], dtype=np.float32), np.array([ 8.29890251,  8.54551315], dtype=np.float32),
                    np.array([ 8.37506676,  8.62712955], dtype=np.float32), np.array([  1.44073129,  18.55874062], dtype=np.float32)]
vector_list_weng = [np.array([ 0.66666669,  0.33333334], dtype=np.float32), np.array([ 1.10131979,  1.75118017], dtype=np.float32),
                    np.array([ 1.2975595 ,  7.89523792], dtype=np.float32), np.array([ 1.31153488,  8.93496418], dtype=np.float32),
                    np.array([  1.41702962,  16.87780952], dtype=np.float32), np.array([  1.43922758,  18.55014038], dtype=np.float32),
                    np.array([  1.4393425 ,  18.55878448], dtype=np.float32)]


graph_weng = []
for i in vector_list_weng:
        differenece = linfDistance([np.array(i)], [np.array(exact)], 'chebyshev')[0,0]
        graph_weng.append(differenece)

graph_ma = []
for i in vector_list_abvi:
        differenece = linfDistance([np.array(i)], [np.array(exact)], 'chebyshev')[0,0]
        graph_ma.append(differenece)

print 'graph_weng', graph_weng
print 'graph ma', graph_ma






