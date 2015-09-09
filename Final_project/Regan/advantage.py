import copy
import itertools
import collections
from operator import add
import scipy
import random
import scipy.cluster.hierarchy as hac
from scipy.spatial import ConvexHull
import numpy as np
import sys
import my_mdp


try:
    from scipy.sparse import csr_matrix, dok_matrix
    from scipy.spatial.distance import cityblock as l1distance
except:
    from sparse_mat import dok_matrix,csr_matrix,l1distance

ftype = np.float32


class Advantage:

    def __init__(self, _mdp, _cluster_error):
        self.mdp = _mdp
        self.cluster_error = _cluster_error

    # def setStateAction(self):
    #     self.n = self.mdp.nstates
    #     self.na = self.mdp.nactions
    #     self.d = self.mdp.d

    def get_initial_distribution(self):
        return self.mdp.initial_states_distribution()

    def clean_Points(self, _points):

        _dic = {}
        for key, value in _points.iteritems():
            if not np.all(value==0):
                _dic[key] = value

        return _dic

    def update(self, dic, _u):
        for k, v in _u.iteritems():
            if isinstance(v, collections.Mapping):
                r = self.update(dic.get(k, {}), v)
                dic[k] = r
            else:
                dic[k] = _u[k]
        return dic

    def cluster_cosine_similarity(self, _Points, _cluster_error):
        d = self.mdp.d
        dic = {}

        Points_dic = self.clean_Points(_Points)
        points_array = np.zeros((len(Points_dic),d), dtype= ftype)
        dic_labels = {}

        counter = 0
        for key, val in Points_dic.iteritems():
            points_array[counter] = val
            dic_labels[counter] = key
            counter += 1

        z = hac.linkage(points_array, metric='cosine', method='complete')

        labels = hac.fcluster(z, _cluster_error, criterion='distance')

        for la in range(1,max(labels)+1):
            dic.setdefault(la, {})

        for index, label in enumerate(labels):
            self.update(dic, {label:{dic_labels[index]:points_array[index, :]} } )

        return dic

    def make_convex_hull(self, _dic, _label):
        #change dictionary types to array and extract lists without their (s,a) pairs
        _points = []

        if _label== 'V':
            for val in _dic.itervalues():
                _points.append(np.float32(val[1]))
        else:
            for val in _dic.itervalues():
                _points.append(np.float32(val))

        _points = np.array(_points)
        try:
            hull = ConvexHull(_points)
            hull_vertices = hull.vertices
            hull_points = _points[hull_vertices, :]
        except scipy.spatial.qhull.QhullError:
            print 'convex hull is not available for label:', _label
            hull_points = _points

        return hull_points

    def get_advantages(self, _clustered_results_val):

        l=[]
        for val in _clustered_results_val.itervalues():
            l.append(val)
        return np.array(l)

    def keys_of_value(self, dct, value):

        for k,v in dct.iteritems():
            if (ftype(v) == ftype(value)).all():
                del dct[k]
                return k

    def justify_cluster(self, _vectors_list, _clustered_pairs_vectors):
        """
        this function get list of vectors of d dimension and a dictionary of pairs and vectors.
        it reassigns pairs to vectors.

        :param _vectors_list: list of d-dimensional vectors
        :param _clustered_pairs_vectors: dictionary of (index, (pairs lists,vectors lists))
        :return: find related pair from _clustered_pairs_vectors to any vector from _convex_hull_vectors_list

        example:
           _clustered_pairs_vectors = {1:[[ 0.        ,  0.10899174],
           [ 0.        ,  0.10899174],
           [ 0.        ,  0.32242826]]}

           _convex_hull_vectors_list =  {1: {(0, 1): array([ 0.        ,  0.10899174], dtype=float32),
             (0, 0): array([ 0.        ,  0.10899174], dtype=float32), (2, 1): array([ 0.        ,  0.32242826], dtype=float32),
             (2, 0): array([ 0.        ,  0.32242826], dtype=float32), (1, 0): array([ 0.        ,  0.01936237], dtype=float32),
             (1, 1): array([ 0.        ,  0.01936237], dtype=float32)}}

            it returns: {1: ([(0, 1), (0, 0), (2, 1)], array([[ 0.        ,  0.10899174],
                       [ 0.        ,  0.10899174],
                       [ 0.        ,  0.32242826]], dtype=float32))}
        """

        _dic_pairs_vectors = {}

        for key in _vectors_list.iterkeys():
            if len(_vectors_list[key]) == 0:
                _dic_pairs_vectors[key] = ([k for k in _clustered_pairs_vectors[key].iterkeys()], self.get_advantages(_clustered_pairs_vectors[key]) )
            else:
                policy = []
                for i in _vectors_list[key]:
                    policy.append(self.keys_of_value(_clustered_pairs_vectors[key], i))
                _dic_pairs_vectors[key] = (policy,_vectors_list[key])

        return _dic_pairs_vectors

    def sum_advantages(self, _Uvec, _points):
        d = self.mdp.d
        sum_result = np.zeros(d)
        for i in _points:
            sum_result = map(add, sum_result, i)

        if all(v == 0 for v in sum_result):
            sum_result = False
        else:
            sum_result = map(add, sum_result, self.get_initial_distribution().dot(_Uvec))

        return sum_result

    def sum_cluster_and_matrix(self, pair_vector_cluster, _matrix_nd):

        """
        this function receives dictionary of clusters including assigned pairs and advantages
        and nxd matrix
        :param pair_vector_cluster:  dictionary of clusters including assigned pairs and advantages in which advantages are vectors of dimension d
        :param _matrix_nd: a related matrix of dimension nxd
        :return: for each cluster if there is (s,a-i) and (s,a_j) choose one of them randomly and make sum on all related vectors in the same cluster
                after add beta.matrix_nd
        """

        n = self.mdp.nstates
        d = self.mdp.d

        final_dic = {}
        dic_clusters_sum_v_old = {}

        for key,val in pair_vector_cluster.iteritems():
            sum_d = np.zeros(d, dtype= ftype)
            pairs_list = []

            for i in range(n):
                selected_pairs = [val[0].index(pair) for pair in val[0] if pair[0]==i]

                if selected_pairs:
                    pair_index = random.choice(selected_pairs)
                    sum_d = map(add, sum_d, val[1][pair_index])
                    pairs_list.append(val[0][pair_index])

            final_dic[key] = (pairs_list, sum_d)

        for k,v in final_dic.iteritems():
            dic_clusters_sum_v_old[k] = (v[0], map(add, self.get_initial_distribution().dot(_matrix_nd), v[1]))

        return dic_clusters_sum_v_old

    def accumulate_advantage_clusters(self, _matrix_nd, _advantages, _cluster_error):

        """

        this function cluster advantages, make a convex hull on each cluster and returns back a dictionary
        of sum of vectors in each cluster and related pair of (state, action) in the same cluster

        :param _matrix_nd: a matrix of dimension nxd that will be added to improvements concluded from advantages
        :param _advantages: set of all generated advantages, each advantage is a vector of dimension d
        :param _cluster_error: max possible distance(cosine similarity distance) between two points in each cluster
        :return: returns back a dictionary of clusters including: key : value. Key is a counter of dictionary
                value is a pair like: ([(0, 1), (2, 0), (0, 0), (2, 1)], [0.0, 0.73071181774139404]) which first element
                are pairs and second element is sum on all related vectors + beta._matrix_nd
        """
        clustered_advantages= self.cluster_cosine_similarity(_advantages, _cluster_error)
        convex_hull_clusters = {}

        for key,val in clustered_advantages.iteritems():
            tempo = self.make_convex_hull(val, key)
            convex_hull_clusters[key] = tempo

        if bool(convex_hull_clusters):
            cluster_pairs_vectors = self.justify_cluster(convex_hull_clusters, clustered_advantages)
            sum_on_convex_hull_temp = self.sum_cluster_and_matrix(cluster_pairs_vectors, _matrix_nd)

            #sum_on_convex_hull_temp = {key: (val[0], self.sum_advantages(_matrix_nd, val[1])) for key, val in cluster_pairs_vectors.iteritems()}
            sum_on_convex_hull = {key:val for key,val in sum_on_convex_hull_temp.iteritems() if val[1]}
            return sum_on_convex_hull

        return {}

    def declare_policies(self, _policies, pi_p):
        """
        this function receives dictionary of state action pairs an related vector value improvements
        and returns back dictionary of policies related to given pairs and the same vector value improvement
        :param _policies: dictionary of this form : {0: ((1, 0), (0, 1)), [ 1.20030463,  0.        ])
        :param pi: the given policy without counting improvement in accounts
        :return: dictionary of new policies and related improved vector values
        """

        _pi_p = pi_p.copy()

        new_policies = {}
        _pi_old = copy.deepcopy(_pi_p)

        for k, policy in _policies.iteritems():
            for key, val in _pi_p.iteritems():
                tempo = [item[1] for item in policy[0] if item[0] == key]
                if tempo:
                    _pi_p[key] = tempo

            new_policies[k] = (_pi_p, np.float32(policy[1]))
            _pi_p = copy.deepcopy(_pi_old)

        return new_policies

def make_test_VVMDP():
        monMdp = my_mdp.VVMdp(
        _startingstate={'buro'},
        _transitions={
            ('buro', 'bouger', 'couloir'): 0.4,
            ('buro', 'bouger', 'buro'): 0.6,
            ('buro', 'rester', 'buro'): 1,
            ('couloir', 'bouger', 'couloir'): 1,
            ('couloir', 'bouger', 'buro'): 0,
            ('couloir', 'rester', 'couloir'): 1,
            ('couloir', 'rester', 'buro'): 0
        },
        _rewards={
            'buro': [1, 0.0],
            'couloir': [1, 0.0]
        }
    )

        monMdp.set_Lambda( [1, 0] )
        return monMdp

# m = make_test_VVMDP()
#
# d = m.d
# na = m.nactions
# n = m.nstates
#
# adv = Advantage(m, 0.1)
#
# state = ({s:[random.randint(0,na-1)] for s in range(n)},
#         np.zeros(d, dtype=ftype) )
#
# Uvec = adv.mdp.value_iteration(epsilon= 0.001, policy=state[0], k=1, _Uvec= state[1], _stationary= True)
# advantage_points_dic = adv.mdp.calculate_advantages_labels(Uvec, state[0], True)
# old_policies = adv.accumulate_advantage_clusters(Uvec, advantage_points_dic, adv.cluster_error)
#new_policies = adv.declare_policies(old_policies, state[0])
#
# print 'new_policies'
# print new_policies
