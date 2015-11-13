import copy
import itertools
import collections
from operator import add
from Queue import Queue
import scipy
import random
import scipy.cluster.hierarchy as hac
from scipy.spatial import ConvexHull
import numpy as np
import sys
from Regan import my_mdp


try:
    from scipy.sparse import csr_matrix, dok_matrix
    from scipy.spatial.distance import cityblock as l1distance
except:
    from sparse_mat import dok_matrix,csr_matrix,l1distance


ftype = np.float32


class no_dominated:
    def __init__(self, _mdp, _lambda):
        self.mdp = _mdp
        self.Lambda = np.zeros(len(_lambda), dtype= ftype)
        self.Lambda[:] = _lambda

    def setStateAction(self):
        self.n = self.mdp.nstates
        self.na = self.mdp.nactions

    def get_initial_distribution(self):
        return self.mdp.initial_states_distribution()

    def clean_Points(self, _points):

        _points_zero_rows = []
        rows = _points.shape[0]

        for r in range(rows):
            if np.all(_points[r,:] == 0):
                _points_zero_rows.append(r)

        counter = 0
        for r in _points_zero_rows:
            _points = np.delete(_points, (r-counter), axis=0 )
            counter+=1

        return _points

    def keys_of_value(self, dct, value):
        for k,v in dct.iteritems():
            if (ftype(v) == ftype(value)).all():
                return k

    def update(self, dic, _u):
        for k, v in _u.iteritems():
            if isinstance(v, collections.Mapping):
                r = self.update(dic.get(k, {}), v)
                dic[k] = r
            else:
                dic[k] = _u[k]
        return dic

    def cluster_cosine_similarity(self, _Points, _cluster_error):

        dic = {}
        Points = self.clean_Points(_Points)

        z = hac.linkage(Points, metric='cosine', method='complete')
        labels = hac.fcluster(z, _cluster_error, criterion='distance')

        for la in range(1,max(labels)+1):
            dic.setdefault(la, {})

        for index, label in enumerate(labels):
            s,a = index/self.na, index%self.na
            self.update(dic, {label:{(s,a):Points[index, :]} } )

        return dic

    def get_advantages(self, _clustered_results_val):

        l=[]
        for val in _clustered_results_val.itervalues():
            l.append(val)
        return np.array(l)

    def justify_cluster(self, _convex_hull_results, _clustered_results):

        _dic = {}

        for key in _convex_hull_results.iterkeys():
            if len(_convex_hull_results[key]) == 0:
                _dic[key] = ([k for k in _clustered_results[key].iterkeys()], self.get_advantages(_clustered_results[key]) )
            else:
                policy = []
                for i in _convex_hull_results[key]:
                    policy.append(self.keys_of_value(_clustered_results[key], i))
                _dic[key] = (policy,_convex_hull_results[key])

        return _dic

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

    def find(self, _V_, _dic):
        _V = np.float32(_V_)
        for key, value in _dic.items():
            val_float32 = np.float32(value[1])
            if all(val_float32 == _V):
                return _dic[key]

        sys.exit("raised an error in searching for the policy")
        return

    def take_dictionary(self, _dic, hull_points):
        output_dic = {}
        for i in range(hull_points.shape[0]):
            output_dic[i] = self.find(hull_points[i,:], _dic)

        return output_dic

    def sum_advantages(self, _Uvec, _points):

        sum_result = np.zeros(len(self.Lambda))
        for i in _points:
            sum_result = map(add, sum_result, i)

        if all(v == 0 for v in sum_result):
            sum_result = False
        else:
            sum_result = map(add, sum_result, self.get_initial_distribution().dot(_Uvec))

        return sum_result

    def accumulate_advantage_clusters(self, _Uvec, _points, _cluster_error):

        clustered_results = self.cluster_cosine_similarity(_points, _cluster_error)
        convex_hull_results = {}

        for key,val in clustered_results.iteritems():
            if val:
                tempo = self.make_convex_hull(val, key)
                if tempo!=[]:
                    convex_hull_results[key] = tempo
                else:
                    convex_hull_results[key] = []

        if bool(convex_hull_results):
            cluster_policies = self.justify_cluster(convex_hull_results, clustered_results)
            sum_on_convex_hull_temp = {key: (val[0], self.sum_advantages(_Uvec, val[1])) for key, val in cluster_policies.iteritems()}
            sum_on_convex_hull = {key:val for key,val in sum_on_convex_hull_temp.iteritems() if val[1]}
            return sum_on_convex_hull

        # plt.scatter(self.Points[:, 0], self.Points[:, 1], c=self.labels.astype(np.float))
        # for val in convex_hull_results.itervalues():
        #     plt.plot( val[:, 0], val[:,1])
        # plt.show()

        return {}

    def take_policy_changes(self, _policies, _Udot_best):
        for val in _policies.itervalues():
            if (ftype(val[1]) == ftype(_Udot_best)).all():
                return val[0]

        sys.exit("raised an error in searching for the policy")
        return

    def declare_policies(self, _policies, pi):
        _pi = pi.copy()

        new_policies = {}
        _pi_old = copy.deepcopy(_pi)

        for k, policy in _policies.iteritems():
            for key, val in _pi.iteritems():
                tempo = [item[1] for item in policy[0] if item[0] == key]
                if tempo:
                    _pi[key] = tempo

            new_policies[k] = (_pi, np.float32(policy[1]))
            _pi = copy.deepcopy(_pi_old)

        return new_policies

    def union_dic(self, _old, _new):

        for values in _new.itervalues():
            length = len(_old)
            _old[length+1] = values

        return _old

    def extend_V(self, val, epsilon, cluster_error):

        Uvec = self.mdp.value_iteration(epsilon= epsilon, policy=val[0], k=1, _Uvec= val[1], _stationary= True)
        advantage_points = self.mdp.calculate_advantages(Uvec, val[0], True)

        old_policies = self.accumulate_advantage_clusters(Uvec, advantage_points, cluster_error)
        new_policies = self.declare_policies(old_policies, val[0])

        return new_policies

    def is_epsilon_dominance(self, _val, _new_policies, _threshold):
        check = False

        for val in _new_policies.itervalues():
            print 'val', val[1]+_threshold
            print '_val', _val[1]
            check = all(np.array(val[1])+ _threshold >= np.array(_val[1]))
            if check:
                return check

        return check

    def calculate_improvement(self, _val, _new_policies):
        impro = 2.2250738585072014e-308

        for val in _new_policies.itervalues():
            tempo = l1distance(val[1], _val[1])
            if tempo > impro:
                impro = tempo
        return impro

    def generate_convex_hull(self, _dic, _policies):

        non_dominated_Udot = self.union_dic(_dic, _policies)
        tempo = self.make_convex_hull(non_dominated_Udot, 'V')
        if tempo.size:
            non_dominated_dic = self.take_dictionary(non_dominated_Udot, tempo)
        else:
            non_dominated_dic = non_dominated_Udot

        return non_dominated_dic

    def is_contain_policy(self, _given_pol, _policy_list):

        my_value = _given_pol[1]
        non_dominated_values = [val[1] for val in _policy_list.itervalues()]

        check = False
        for v in _policy_list.itervalues():
            if (v[1]== my_value).all():
                check = True
        return check

    def generate_non_dominated_V(self, epsilon=0.1, cluster_error=0.1, threshold = 0.1):
        """
        This function returns back an approximated set of non-dominated vectors for a mdp according to three given
        accuracy thresholds
        :param epsilon:
        :param cluster_error: the error of generating clusters on advantages
        :param threshold:
        :return:
        """

        log = open("tests.txt", "w")

        d = self.mdp.d
        gamma, R = self.mdp.gamma , self.mdp.rewards

        improvement = sys.float_info.max
        non_dominated_Udot_dic = {}

        Uvec_old = np.zeros( (self.n, d), dtype=ftype)
        Udot_old = np.zeros(d, dtype=ftype)

        #start with a random policy
        pi = {s:[random.randint(0,self.na-1)] for s in range(self.n)}
        non_dominated_Udot_dic[0] = (pi, Udot_old)

        non_dominated_Udot_queue = Queue()
        non_dominated_Udot_queue.put((pi, Udot_old), block=True, timeout=None)

        while not non_dominated_Udot_queue.empty():
            val = non_dominated_Udot_queue.get(block=True, timeout=None)

            print >> log,'val', val
            print >> log, 'non_dominated_Udot_dic.values()',non_dominated_Udot_dic.values()

            if self.is_contain_policy(val, non_dominated_Udot_dic):
                print >> log, 'True'
                new_policies = self.extend_V(val, epsilon, cluster_error)

                print >> log,'new_policies', new_policies
                improvement = self.calculate_improvement(val, new_policies)
                print >> log,'improvement', improvement

                #if improvement<threshold :
                if not self.is_epsilon_dominance(val, new_policies, threshold):

                    print >> log, '99999999999999999999999999999999999999999999999'
                    print >> log, 'final improvement', improvement

                    non_dominated_Udot_dic = self.generate_convex_hull(non_dominated_Udot_dic, new_policies)
                    print >> log, "final final answer"
                    print >> log, non_dominated_Udot_dic

                    print 'final nondominated dictionary'
                    print non_dominated_Udot_dic
                    return non_dominated_Udot_dic

                for value in new_policies.itervalues():
                    non_dominated_Udot_queue.put(value, block=True, timeout=None)

                non_dominated_Udot_dic = self.generate_convex_hull(non_dominated_Udot_dic, new_policies)

            else:
                print >> log,'false'

            print>> log, '****************'

        print >> log, "final final answer"
        print >> log, non_dominated_Udot_dic
        return non_dominated_Udot_dic