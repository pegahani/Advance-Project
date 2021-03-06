import copy
import advantage
import utils
import numpy as np

try:
    from scipy.sparse import csr_matrix, dok_matrix
    from scipy.spatial.distance import cityblock as l1distance
    from scipy.spatial.distance import cdist as linfDistance
except:
    from sparse_mat import dok_matrix,csr_matrix,l1distance

ftype = np.float32


class Problem:
    """The abstract class for a formal problem.  You should subclass this and
    implement the method successor, and possibly __init__, goal_test, and
    path_cost. Then you will create instances of your subclass and solve them
    with the various search functions."""

    def __init__(self, initial, _mdp, _cluster_error, _epsilon):
        """The constructor specifies the initial state, and possibly a goal
        state, if there is a unique goal.  Your subclass's constructor can add
        other arguments."""
        self.initial = initial
        self.cluster_error = _cluster_error
        self.epsilon = _epsilon
        self.mdp = _mdp

        self.adv = advantage.Advantage(_mdp, _cluster_error)
        #self.cluster_numbers = 0

    def actions(self, state):
        """
        this function takes an state like [{pi_p,v_bar_d,matrix_nd}] as an example
        :param state: [{0: [0], 1: [0], 2: [1], 3: [1], 4: [1]}, array([ 0.,  0.], dtype=float32), array([[ 0.,  0.],
               [ 0.,  0.],
               [ 0.,  0.],
               [ 0.,  0.],
               [ 0.,  0.]], dtype=float32)]
        :return: it returns dictionary of policies and related V_bars which are results of clustering on advantages for a given state
                as an example : {1: ({0: [0], 1: [0, 1], 2: [0, 1], 3: [1], 4: [1]}, array([ 1.54556584,  0.        ], dtype=float32)),
                                2: ({0: [1, 0], 1: [0], 2: [1], 3: [0, 1], 4: [1, 0]}, array([ 0.       ,  3.2635324], dtype=float32))}
        """

        advantages_pair_vector_dic = self.adv.mdp.calculate_advantages_labels(state[2], True)
        cluster_advantages = self.adv.accumulate_advantage_clusters(state[2], advantages_pair_vector_dic, self.adv.cluster_error)
        policies = self.adv.declare_policies(cluster_advantages, state[0])

        return policies

    def add_equal_matrixes_nd(self, policy_v_bar_dic, state):
        """
        this function receives dictionary of policies and v_bar_ds, the current state
        and it returns back dictionary of same policies ,v_bar_d vectors and related matrix_nd using their parent: state
        :param policy_v_bar_dic:dictionary of policies and V_bars
        :param state: is a parent state of policy_v_bar_dic, (each memeber of dictionary will be a child state of parent)
        :return: it returns the same dictionary, with a new matrix_nd for each member. this new matrix is the equal matrix
        for V_bar inside policy_v_bar_dic
        """

        policy_v_bar_matrix_dic = {}
        l = 0

        for va in policy_v_bar_dic.itervalues():
            #uvec = self.adv.mdp.value_iteration(epsilon=0.001, policy=val[0], k=1, _Uvec= state[2], _stationary= True)
            uvec = self.adv.mdp.update_matrix(policy_p=va[0],_Uvec_nd= state[2])

            temp = list(va)
            temp.append(uvec)
            policy_v_bar_matrix_dic[l] = temp

            l+=1

        return policy_v_bar_matrix_dic

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

    def successor(self, state):
        """Given a state, return a sequence of (action, state) pairs reachable
        from this state. If there are many successors, consider an iterator
        that yields the successors one at a time, rather than building them
        all at once. Iterators will work fine within the framework."""
        abstract

    def goal_test(self, improvement):
        """Return True if the state is a goal. The default method compares the
        state to self.goal, as specified in the constructor. Implement this
        method if checking against a single self.goal is not enough."""
        #return state == self.goal
        return improvement < self.epsilon

    def path_cost(self, c, state1, action, state2):
        """Return the cost of a solution path that arrives at state2 from
        state1 via action, assuming cost c to get up to state1. If the problem
        is such that the path doesn't matter, this function will only look at
        state2.  If the path does matter, it will consider c and maybe state1
        and action. The default method costs 1 for every step in the path."""
        return c + 1

    def value(self):
        """For optimization problems, each state has a value.  Hill-climbing
        and related algorithms try to maximize this value."""
        abstract

class Node:
    """A node in a search tree. Contains a pointer to the parent (the node
    that this is a successor of) and to the actual state for this node. Note
    that if a state is arrived at by two paths, then there are two nodes with
    the same state.  Also includes the action that got us to this state, and
    the total path_cost (also known as g) to reach the node.  Other functions
    may add an f and h value; see best_first_graph_search and astar_search for
    an explanation of how the f and h values are handled. You will not need to
    subclass this class."""

    def __init__(self, state, parent=None, action=None, path_cost=0, improvement = 1000.0, inside_convex= False):
        "Create a search tree Node, derived from a parent by an action."
        utils.update(self, state=state, parent=parent, action=action,
               path_cost=path_cost, depth=0, improvement= improvement, inside_convex= inside_convex)
        self.state = state
        if parent:
            self.parent = parent
            self.depth = parent.depth + 1
            self.improvement = linfDistance([np.array(self.state[1])], [np.array(parent.state[1])], 'chebyshev')[0,0]
            #self.improvement = linfDistance([np.array(self.state[1])], [np.array(parent.state[1])], 'cityblock')[0,0]


    def update_state(self, uvec):
        sta = self.state
        sta[2] = np.array(uvec, dtype=ftype)
        self.state = sta

    def __repr__(self):
        return "<Node %s>" % (self.state,)

    def child_node(self, problem, _state):
        """
        this function recieves problem and a given state and generates related node for the given state
        :param problem: the general problem
        :param _state:  state is set of policy, v_bar of dimension d, and a nxd equal matrix
        as an example:
            [{0: [0], 1: [1], 2: [0], 3: [0], 4: [1]},array([ 6.46138573,  0.        ], dtype=float32), array([[ 1.57669592,  0.        ],
           [ 1.33064532,  0.        ],
           [ 0.95161259,  0.        ],
           [ 1.18965316,  0.        ],
           [ 1.41277897,  0.        ]], dtype=float32)]
        :return: it returns a node with given state, _state. and the node has an action= None
        """

        state = _state
        action = None

        node = Node(state, self, action,
                    problem.path_cost(self.path_cost, self.state, action, state))

        return node

    def expand(self, problem):
        """
        for the current problem with current state self.state
        this function produce all current state children and returns back list of children nodes
        :param problem: the problem
        :return: set of children node for current state self.state :
        each state is a list of three elements : [policy_dictionary, v_bar_vector_d, matrix_nd]

        return example is like: for d=2 and states = 5
            [<Node [{0: [1], 1: [1], 2: [1], 3: [0], 4: [1]}, array([ 2.90815639,  2.74776697], dtype=float32), array([[ 0.11989779,  0.26664305],
           [ 0.2361128 ,  1.37656164],
           [ 1.06986642,  1.20945454],
           [ 1.70635402,  1.13206553],
           [ 0.94032305,  0.4999688 ]], dtype=float32)]>, <Node [{0: [0], 1: [1], 2: [0], 3: [0], 4: [1]}, array([ 2.91628027,  3.01785111], dtype=float32), array([[ 0.23656178,  0.28704047],
           [ 0.2361128 ,  1.37656164],
           [ 1.03994453,  1.12303126],
           [ 1.67879081,  1.05245483],
           [ 0.94032305,  0.4999688 ]], dtype=float32)]>, <Node [{0: [0], 1: [0], 2: [0], 3: [0], 4: [1]}, array([ 3.17764711,  2.82243824], dtype=float32), array([[ 0.23656178,  0.28704047],
           [ 0.36672422,  1.09213078],
           [ 1.14322567,  0.89811742],
           [ 1.77393055,  0.84527034],
           [ 0.97343946,  0.42785156]], dtype=float32)]>, <Node [{0: [0], 1: [1], 2: [0], 3: [0], 4: [1]}, array([ 3.16436195,  2.69948459], dtype=float32), array([[ 0.23656178,  0.28704047],
           [ 0.2361128 ,  1.37656164],
           [ 1.03994453,  1.12303126],
           [ 1.67879081,  1.05245483],
           [ 0.94032305,  0.4999688 ]], dtype=float32)]>, <Node [{0: [0], 1: [1], 2: [0], 3: [1], 4: [1]}, array([ 3.06681347,  2.50243473], dtype=float32), array([[ 0.23656178,  0.28704047],
           [ 0.2361128 ,  1.37656164],
           [ 1.03994453,  1.12303126],
           [ 1.81535518,  0.7132594 ],
           [ 0.94032305,  0.4999688 ]], dtype=float32)]>]
        """

        "List the reachable  nodes in one step from the given state."
        policy_vector = problem.actions(self.state)
        policy_vector_matrix = problem.add_equal_matrixes_nd(policy_vector, self.state)

        return [self.child_node(problem, val)
                for val in policy_vector_matrix.itervalues()]

    def solution(self):
        "Return the sequence of actions to go from the root to this node."
        return [node.action for node in self.path()[1:]]

    def path(self):
        "Return a list of nodes forming the path from the root to this node."
        node, path_back = self, []
        while node:
            path_back.append(node)
            node = node.parent
        return list(reversed(path_back))

    # We want for a queue of nodes in breadth_first_search or
    # astar_search to have no duplicated states, so we treat nodes
    # with the same state as equal. [Problem: this may not be what you
    # want in other contexts.]

    def __eq__(self, other):
        return isinstance(other, Node) and self.state == other.state

    def __hash__(self):
        return hash(self.state)
