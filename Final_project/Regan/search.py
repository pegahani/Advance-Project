import time
import random
import Problem
import propagation_V
import V_bar_search
import my_mdp
import numpy as np
from itertools import chain


try:
    from scipy.sparse import csr_matrix, dok_matrix
    from scipy.spatial.distance import cityblock as l1distance
    from scipy.spatial.distance import cdist as linfDistance
except:
    from sparse_mat import dok_matrix,csr_matrix,l1distance

ftype = np.float32

#____________________________________________________________________________


def interior_easy_points(dim):
    #dim = len(self.points[0])
    l = []
    for i in range(dim):
        l.append(random.uniform(0.0, 1.0))
    return l

def dot_product(lista, listb):
    return [a*b for a,b in zip(lista,listb)]


# if __name__ == '__main__':
#
#     log2 = open("A_star_response" + ".txt", "w")
#
#     n= 5
#     na = 2
#     d= 2
#
#     _lambda_rand = interior_easy_points(d)
#     _r = my_mdp.generate_random_reward_function(n, na, d)
#     m = my_mdp.make_simulate_mdp(n, na, _lambda_rand, _r)
#
#     Uvec = m.policy_iteration()
#     exact = m.initial_states_distribution().dot(Uvec)
#     print >> log2, '-----------------------'
#     log2.flush()
#     print >> log2, "exact response", exact
#     log2.flush()
#
#     p = Problem.Problem(initial= [{s:[random.randint(0,na-1)] for s in range(n)}, np.zeros(d, dtype=ftype), np.zeros((n,d),dtype=ftype)],
#                  _mdp=m, _cluster_error= 0.1, _epsilon=0.00001)
#
#     v_prog = propagation_V.propagation_V(m= m, d = d, cluster_v_bar_epsilon = 0.001)
#     V_vectors = v_prog.convex_hull_search_better(prob=p, iteraion_number=1000)
#     # print "V_vectors", V_vectors
#
#     #concatenated = chain(range(1, 101, 10), range(1000, 10000, 1000), [9999])
#
#     #for i in concatenated:
#     V = V_bar_search.V_bar_search(_mdp= m, _V_bar=V_vectors, lam_random= m.get_lambda())
#     v_opt = V.v_optimal()
#
#     print >> log2, "V_bar_list", V.V_bar_list_d
#     log2.flush()
#
#     print >> log2, 'v_optimal', v_opt
#     log2.flush()
#
#     print >> log2, 'number of asked queries from user', V.query_number
#
#     print >> log2, 'random lambda after all', m.get_lambda()
#     t1 = np.dot(v_opt, np.array(_lambda_rand))
#     t2 = np.dot(exact, np.array(_lambda_rand))
#     print >> log2, 'final answer after multiplying lambda', t1
#     print >> log2, 'exact answer after multiplying lambda', t2
#
#     print >> log2, "**************difference for iteration************** :", abs(t1-t2)
#     log2.flush()


if __name__ == '__main__':

    log1 = open("exact_response" + ".txt", "w")
    na = 5
    d = 2

    average_on = 10

    iterate_v_propagation = 100
    _cluster_v_bar_epsilon = 0.001


    for n in range(2,5,2):

        time_gather_V_bars = []
        time_find_V_optimal = []
        number_asked_queries = []
        error = []

        for iteration in range(average_on):

            _lambda_rand = interior_easy_points(d)
            _r = my_mdp.generate_random_reward_function(n, na, d)
            m = my_mdp.make_simulate_mdp(n, na, _lambda_rand, _r)

            Uvec = m.policy_iteration()
            exact = m.initial_states_distribution().dot(Uvec)

            p = Problem.Problem(initial= [{s:[random.randint(0,na-1)] for s in range(n)}, np.zeros(d, dtype=ftype), np.zeros((n,d),dtype=ftype)],
                         _mdp=m, _cluster_error= 0.1, _epsilon=0.00001)
            v_prog = propagation_V.propagation_V(m= m, d = d, cluster_v_bar_epsilon = _cluster_v_bar_epsilon)

            start_time_gather_V_bars = time.time()
            V_vectors = v_prog.convex_hull_search_better(p, iterate_v_propagation)
            time_gather_V_bars.append(time.time() - start_time_gather_V_bars)

            V = V_bar_search.V_bar_search(_mdp= m, _V_bar=V_vectors, lam_random= m.get_lambda())

            start_time_find_V_optimal = time.time()
            v_opt = V.v_optimal()
            time_find_V_optimal.append(time.time() - start_time_find_V_optimal)

            number_asked_queries.append(V.query_number)

            t1 =  np.dot( v_opt, np.array(_lambda_rand) )
            t2 = np.dot( exact, np.array(_lambda_rand) )
            error.append(abs(t1-t2))



        print >> log1, '-----------------------'
        print >> log1, iterate_v_propagation,'-th', 'iteration for |states|=', n
        log1.flush()

        print >> log1,'queries=', np.mean(number_asked_queries)
        log1.flush()

        print >> log1,'time_gather_V_bars =', np.mean(time_gather_V_bars)
        log1.flush()

        print >> log1,'time_find_V_optimal =', np.mean(time_find_V_optimal)
        log1.flush()

        print >> log1,'error =', np.mean(error)
        log1.flush()








