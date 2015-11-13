import copy
import random
import scipy
import Problem
import numpy as np
import utils
import scipy.spatial
import advantage
from scipy.spatial import ConvexHull
import scipy.cluster.hierarchy as hac
from scipy.spatial import Delaunay

from scipy.spatial.distance import cdist as linfDistance


ftype = np.float32

class propagation_V:

    def __init__(self, m, d, cluster_v_bar_epsilon, epsilon_error):
        self.m = m
        self.d = d
        self.cluster_v_bar_epsilon = cluster_v_bar_epsilon
        self.epsilon_error = epsilon_error

    def explored_contain(self, child, _explored):
        check = False
        for val in _explored.itervalues():
            if (val[1]== child.state[1]).all():
                check = True
        return check

    def inside_selected_frontier(self, _node, selected_frontier):
        check = False
        for val in selected_frontier.A:
            if (_node.state[2] == val.state[2]).all() or (abs(_node.state[2] - val.state[2]) <=  0.01).all():
                check = True
                return check

        return check

    def epsilon_dominance_vector(self, _u, _v, _epsi):
        return all( (1+_epsi)*_u >= _v )

    def epsilon_dominant_check(self, _child, _explored, _epsi):
        """
        get child node and dictionary of node.state s and it returns back state key dominated to given node _child
        :param _child: is a node type as an example: <Node [{0: [1], 1: [1], 2: [1], 3: [1], 4: [0]}, array([ 0.,  0.], dtype=float32), array([[ 0.,  0.],
                                                           [ 0.,  0.],
                                                           [ 0.,  0.],
                                                           [ 0.,  0.],
                                                           [ 0.,  0.]], dtype=float32)]>
        :param _explored: is a dictionary of nodes for instance: {0: node.state, 1: node.state, 2: node.state, 3:node.state}
        :param _epsi:is the epsilon value that is used for epsilon-dominated check
        :return: returns -1 if there is no dominated element in explored to _child, either it returns dictionary value of node.state.V-bar dominated to _child.state[1]
        """

        for key, val in _explored.iteritems():
            if self.epsilon_dominance_vector(val[1], _child.state[1], _epsi):
                return key
        return -1

#****************** ****** ******************
    def _vector_inside(self, _u_d, _v_d):
        """
        for two given vectors, it checks if they are equal or not
        :param _u:
        :param _v:
        :return: returns True, if they are equal, either it returns False.
        """
        return all(_u_d == _v_d)

    def explored_contain(self, _child, _explored):

        """
        it checks whether child is inside explore dictionary of states or not
        :param _child: a given node
        :param _explored: dictionary of states as {0:[policy,vector_d,matrix_nd], 1:[policy,vector_d,matrix_nd], 2:[policy,vector_d,matrix_nd]}
        :return: it returns value of dictionary, if _child exists in _explored. either it returns -1
        """

        for key, val in _explored.iteritems():
            if self._vector_inside(_child.state[1], val[1]):
                return key
        return -1
#****************** ****** ******************

    def update_explored_epsilon_dominance_check(self, node, explored, epsilon):
        """
        this function receives a node and set of nodes' states. it checks if there is a epsilon-dominated V_bar inside explored dictionary to the given node.state[1]
        if there isn't such a member, function add new element to explored dictionary, either it does nothing.
        *the secon element is state is the equal V_bar
        :param node: is a node type as an example: <Node [{0: [1], 1: [1], 2: [1], 3: [1], 4: [0]}, array([ 0.,  0.], dtype=float32), array([[ 0.,  0.],
                                                           [ 0.,  0.],
                                                           [ 0.,  0.],
                                                           [ 0.,  0.],
                                                           [ 0.,  0.]], dtype=float32)]>
        :param explored: is a dictionary of nodes for instance: {0: node.state, 1: node.state, 2: node.state, 3:node.state}
        :param epsilon: is the epsilon value that is used for epsilon-dominated check
        :return: it returns update explored dictionary, either is the same either new node will be added as a new member
        """
        index_epsilon_dominance = self.epsilon_dominant_check(node, explored, epsilon)
        "if there is no epsilon-dominated element inside explored set to the given node, add this new V to the explored"
        if index_epsilon_dominance == -1:
            l = len(explored)
            explored.update({l:node.state})

        return explored

    def get_v_(self, explored):

        result = []
        for val in explored.itervalues():
            result.append(val[1])
        return result

    def get_v_states(self, _frontier):

        result = []
        for val in _frontier:
            result.append(val.state[1])
        return result

    def produce_policy(self, v_n):
        """
        it takes an array form of policy and transforms it to dictionary form of policy
        :param v_n: an array of dimension |states|
        :return: a dictionary form of policy
        """
        return {i:[v_n[i]] for i in range(len(v_n))}

    def get_frontier_convex_hull_vertices(self, frontier_A_d):

        """
        it gets a list of nodes, extracts list of its d_dimensional vectors, then it adds zero vector to the list.
        it finds a convex hull of these vectors. finally it returns vertices of the obtained convex hull.
        output is list of vector indexes inside the generated list: list of d_dimensional v_vectors plus d-dimensional zero vector.

        :param frontier_A_d: list of nodes. this is example of frontier_A_d :
             [<Node [{0: [0], 1: [2]}, array([ 8.1744051,  0.       ], dtype=float32), array([[ 8.01563644,  0.        ],
                [ 8.33317375,  0.        ]], dtype=float32)]>, <Node [{0: [0], 1: [0]}, array([ 9.8657198,  0.       ], dtype=float32), array([[ 9.74746609,  0.        ],
                [ 9.9839735 ,  0.        ]], dtype=float32)]>]
        :return: list of indexes of frontier_A_d.A which are inside the generated convex hull.
        """

        list_v_bars = [] #it keeps list of V_bar vectors from frontier_A_d
        zero_vector = np.zeros(self.d, dtype= ftype)

        for item in frontier_A_d:
            item.inside_convex = True
            list_v_bars.append(item.state[1])

        list_v_bars.append(zero_vector)
        vectors_array = np.array(list_v_bars)

        try:
            hull = ConvexHull(vectors_array)
            hull_vertices = list(hull.vertices)
        except scipy.spatial.qhull.QhullError:
            print 'convex hull is not available for this set of vectors'
            hull_vertices = [i for i in range(len(list_v_bars))]

        return hull_vertices

    def random_index_each_cluster(self, clustered_lables, indexes):
        """
        this function gets two lists
        :param clustered_lables: list of indexes of clusters on frontier.A V_bar members
        :param indexes: list of indexes of frontier.A V_bar members
        :return: choose a random index from each cluster and return back it's indexes inside indexes
        """
        retrieve_convex_hull = []
        for j in range(1,max(clustered_lables)+1, 1):
            get_indexes = [i for i in range(0,len(clustered_lables)) if clustered_lables[i]==j]
            retrieve_convex_hull.append(indexes[random.choice(get_indexes)])

        return retrieve_convex_hull

    def frontier_convex_hull(self, frontier):
        """
        gets frontier from our_data_struc type and returns back vertices of the generated convex hull on frontier V_bar
        members. in fact, it returns back indexes of these vertices.
        :param frontier: our_data_struc type
        :return: indexes of frontier.A inside the modified convex hull after get rid of very close V_bar vectors
        """
        print "****************************************"

        cluster_v_bar_epsilon = self.cluster_v_bar_epsilon
        index_list = self.get_frontier_convex_hull_vertices(frontier.A)

        index_list.remove(len(frontier.A))

        Points = []
        for item in index_list:
            Points.append(frontier.A[item].state[1])

        points_array = np.array(Points)

        z = hac.linkage(points_array, metric='cosine', method='complete')
        labels = hac.fcluster(z, cluster_v_bar_epsilon, criterion='distance')

        improve_convex_hull = self.random_index_each_cluster(labels, index_list)

        return improve_convex_hull

    def array_index(self, given_array, array_list):
        """
        it finds array index in list of arrays and return its index
        :param given_array: array of dimension d
        :param array_list: list of arrays of dimension d
        :return: index of existed array in list of arrays
        """

        for i in range(len(array_list)):
            if np.array_equal(array_list[i], given_array):
                return  i

        return -1

    def fill_explored(self, node_list):
        """
        it receives list of Nodes and put them in the dictionary
        :node_list: list of Nodes
        :return: dictionary namely explored
        """
        explored = {}
        for j in range(len(node_list)):
            explored[j] = node_list[j].state[1]

        return explored

    def convex_hull_search_better(self, prob, iteraion_number):
        #keep_vector_results = []

        m = self.m
        obs = open("observe-search" + ".txt", "w")

        frontier = utils.my_data_struc()
        for i in range(self.d):
            m.set_Lambda(np.array([1 if j==i else 0 for j in xrange(self.d)]))
            Uvec_n_d = m.policy_iteration()
            v_d = m.initial_states_distribution().dot(Uvec_n_d)
            n = Problem.Node([self.produce_policy(m.best_policy(Uvec_n_d)), v_d, Uvec_n_d])
            frontier.append(n)

        index_list = self.frontier_convex_hull(frontier)
        frontier.update(index_list)

        explored = self.fill_explored(frontier.A)
        print >> obs, 'explored members', [val for val in explored.itervalues()]

        #best_improve_for_node = 1000.0
        #iteration = 0
        #after first iteration-------------------
        #while best_improve_for_node > prob.epsilon :
        for iteration in range(iteraion_number):
            #print >> obs, 'iteration ------------------', iteration
            #print >> obs, 'best improvement', best_improve_for_node
            obs.flush()
            #iteration += 1

            frontier_addition = utils.my_data_struc()

            #max_improvement_list = []
            for node in frontier.A:
                #max_improvement = -100.0
                for child in node.expand(problem= prob):
                    frontier_addition.append(child)
                    improv_new = child.improvement
                    #if improv_new > max_improvement:
                    #    max_improvement = improv_new

                #max_improvement_list.append(max_improvement)

            #best_improve_for_node = max(max_improvement_list)

            for node in frontier_addition.A:
                frontier.append(node)

            index_list = self.frontier_convex_hull(frontier)
            frontier.update(index_list)
            explored = self.fill_explored(frontier.A)

            explored_list = [val for val in explored.itervalues()]
            #print >> obs, 'explored members', explored_list
            #keep_vector_results.append(explored_list)

        #return  keep_vector_results
        #print >> obs, 'explored members final', [val for val in explored.itervalues()]
        return [val for val in explored.itervalues()]


    def make_convex_hull(self, P_initial, hull_vertices):
        """
        makes convex hull of given Points
        :param P_initial: matrix of many d-dimensional rows
        :return: pair of updated P_initial and list of vertices row index in P_initial
        """

        try:
            hull = ConvexHull(P_initial)
            hull_vertices = list(hull.vertices)
            P_initial = P_initial[hull_vertices, :]
        except scipy.spatial.qhull.QhullError:
            print 'convex hull is not available'
            P_initial = P_initial
            hull_vertices = hull_vertices

        if not (all(P_initial[0, :] == [0] * self.d)):
            P_initial = np.insert(P_initial, 0, np.zeros(shape=(1, self.d)), 0)
            hull_vertices.append(0)

        return (P_initial, hull_vertices)

    def in_hull(self, p, hull):
        """
        Test if point `p` is in `hull`

        `p` should be a `NxK` coordinates of `N` points in `K` dimensions
        `hull` is either a scipy.spatial.Delaunay object or the `MxK` array of the
        coordinates of `M` points in `K`dimensions for which Delaunay triangulation
        will be computed
        """
        q = np.array([p])

        if not isinstance(hull,Delaunay):
            try:
                hull = Delaunay(hull)
            except scipy.spatial.qhull.QhullError:
                return False

        return hull.find_simplex(q)>=0

    def epsilon_close_convex_hull(self, V_d, P_inintial, epsilon):
        """
        the function gets vector and check if there is a vector inside P_initial which is epsilon close to V_d
        :param V_d: d dimensional vector
        :param P_inintial: array of d dimensional rows
        :return: True or False
        """
        for item in xrange(P_inintial.shape[0]):
            dist = linfDistance([np.array(P_inintial[item, :])], [np.array(V_d)], 'chebyshev')[0,0]
            if dist < epsilon:
                return True

        return False

    def check_epsilon(self, V_d, P_intial, epsilon):
        """
        this function gets a new d dimensional V_d and checks if V_d is inside Conv(P_initial) or
        is it p in P_initial in which ||p-V_d|| <= epsilon and returns True. if non of these situations satisfy,it
        returns false.
        :param V_d: a d dimensional vector
        :param P_intial: array of several d-dimensional vectors
        :param epsilon: the epsilon error for checking
        :return: True or False
        """

        #if vector is inside old convex hull
        if self.in_hull(V_d, P_intial):
            return True
        #if vector is epsilon close to a vector in P_initial
        if self.epsilon_close_convex_hull(V_d, P_intial, epsilon):
            return True

        return False

    def update_convex_hull_epsilon(self, P_initial, frontier, hull_vertices, problem):
        """
        this function gets set of current polytope vertices, generates new points using clustering on advantages
        and make a new convex hull of them.
        :param P_initial: matrix of d-dimensional rows
        :param frontier: queue of type my_data_struc includes all nodes for extension
        :return: pairs of (P_initial, frontier) in which P_initial includes 0 vector.
        """

        frontier_addition = utils.my_data_struc()
        P_new = P_initial

        for node in frontier.A:
            for child in node.expand(problem= problem):
                if not (self.check_epsilon(child.state[1], P_initial, self.epsilon_error)):
                    frontier_addition.append(child)
                    P_new = np.vstack([P_new, child.state[1]])

        length_hull_vertices = len(hull_vertices)
        counter = 0
        for node in frontier_addition.A:
            frontier.append(node)
            hull_vertices.append(length_hull_vertices+counter)
            counter += 1

        temp_convex = self.make_convex_hull(P_new, hull_vertices)
        P_initial = temp_convex[0]
        hull_vertices = temp_convex[1]

        frontier.update([item-1 for item in hull_vertices if item-1 >= 0])

        return (P_initial, frontier, hull_vertices)

    def convex_hull_search(self, prob):
        """
        this function gets a problem as tree of nodes each node is pair of policy and V_bar as ({0:2, 1:0, 2:1, 3:2, 4:1}, [0.1,0.25])
        and try to propagate all v_bar using extending node in each iteration and take their vertices of the optimal convex hull.
        :param problem: tree of nodes each node is pair of policy and V_bar as ({0:2, 1:0, 2:1, 3:2, 4:1}, [0.1,0.25])
        :return: returns set of approximated non-dominated v_bar vectors: vectors of dimension d
        """

        P_initial= np.zeros(shape=(1, self.d))
        m = self.m
        frontier = utils.my_data_struc()

        for i in range(self.d):
            m.set_Lambda(np.array([1 if j==i else 0 for j in xrange(self.d)]))
            Uvec_n_d = m.policy_iteration()
            v_d = m.initial_states_distribution().dot(Uvec_n_d)
            n = Problem.Node([self.produce_policy(m.best_policy(Uvec_n_d)), v_d, Uvec_n_d])
            frontier.append(n)

        for item in range(self.d):
            P_initial = np.vstack([P_initial, frontier.A[item].state[1]])

        hull_vertices = range(self.d + 1)

        #make convex hull of points *******************************
        temp_convex = self.make_convex_hull(P_initial, hull_vertices)
        P_initial = temp_convex[0]
        hull_vertices = temp_convex[1]
        #************************************************************
        frontier.update([item-1 for item in hull_vertices if item-1 >= 0])

        temp = self.update_convex_hull_epsilon(P_initial, frontier, hull_vertices, prob)
        P_new = temp[0]
        frontier = temp[1]
        hull_vertices = temp[2]

        #while P_initial and P_new are not equal
        while not (np.array_equal(P_initial, P_new)):
        #for i in range(1000):
            P_initial = P_new
            temp = self.update_convex_hull_epsilon(P_initial, frontier, hull_vertices, prob)
            P_new = temp[0]
            frontier = temp[1]
            hull_vertices = temp[2]

        print 'P_new', P_new
        return [val for val in P_new[1:]]
