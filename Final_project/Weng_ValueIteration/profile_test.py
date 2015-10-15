from ValueIteration import *


if __name__ == '__main__':

    _state, _action, _d = 3, 2, 2

    _Lambda_inequalities = generate_inequalities(_d)
    _lambda_rand = interior_easy_points(_d)

    print "_lambda_rand", _lambda_rand

    #_r = my_mdp.generate_random_reward_function(_state, _action, _d)
    #m = my_mdp.make_simulate_mdp(_state, _action, _lambda_rand,None)
    m = my_mdp.make_simulate_mdp_Yann(_state, _action, _lambda_rand,None)
    print 'm', m

    print 'm.rewards', m.rewards
    print 'm.transitions', m.transitions

    # w = Weng(m, _lambda_rand, _Lambda_inequalities)
    # w.setStateAction()
    #
    # print w.value_iteration_with_advantages(_epsilon=0.001, k=100000, noise= None, cluster_error = 0.01, threshold = 0.0001)
    #
    # m.set_Lambda(_lambda_rand)
    # Uvec = m.policy_iteration()
    # exact = m.initial_states_distribution().dot(Uvec)
    #
    #
    # print 'exact', exact
