import ValueIteration
import pylab as plt

import sys
sys.path.insert(0, '/Users/pegah/Dropbox/GitHub-Projects/Advance-Project/Final_project')

import Regan.my_mdp


if __name__ == '__main__':



    _d = 2

    _Lambda_inequalities = ValueIteration.generate_inequalities(_d)
    #_lambda_rand = ValueIteration.interior_easy_points(_d)

    _lambda_rand = [0.1, 0.5]

    print "_Lambda_rand", _lambda_rand
    print "type(_Lambda_rand)", type(_lambda_rand)

    m = Regan.my_mdp.make_grid_VVMDP()
    #m = Regan.my_mdp.make_test_VVMDP()

    _state, _action = 4, 5
    #m = Regan.my_mdp.make_simulate_mdp_Yann(_state, _action, _lambda_rand, None)

    # print "*******"
    # print "m.actions", m.actions
    # print "m.states", m.states

    w = ValueIteration.Weng(m, _lambda_rand, _Lambda_inequalities)
    w.setStateAction()

    m.set_Lambda(_lambda_rand)
    Uvec = m.policy_iteration()
    exact = m.initial_states_distribution().dot(Uvec)

    print 'exact', exact

    output_ma = w.value_iteration_with_advantages(_epsilon=0.001, k=100000, noise= None,
                                                  cluster_error = 0.0001, threshold = 0.0001, exact= exact)

    _Lambda_inequalities = ValueIteration.generate_inequalities(_d)
    w.reset(m, _lambda_rand, _Lambda_inequalities)

    #output_paul = w.value_iteration_weng(k=100000, noise= None, threshold=0.0001, exact = exact)


    #print '--------------------'
    #print 'approximation', output[4]
    #print 'exact', exact

    #print 'last error weng', output_paul[3][len(output_paul[3])-1]
    #print 'last error ours', output_ma[3][len(output_ma[3])-1]

    print 'error ma', sum(a*b for a,b in zip(list(_lambda_rand), list(output_ma[4]))) - \
                         sum(a*b for a,b in zip(list(_lambda_rand), list(exact)))

    #print 'error paul', sum(a*b for a,b in zip(list(_lambda_rand), list(output_paul[4]))) - \
    #                      sum(a*b for a,b in zip(list(_lambda_rand), list(exact)))

    #print 'last query weng', output_paul[2][len(output_paul[2])-1]
    print 'last query ours', output_ma[2][len(output_ma[2])-1]

    ax = plt.subplot(111)
    ax.plot(output_ma[2], output_ma[3],'r')
    #ax.plot(output_paul[2], output_paul[3],'b')
    plt.show()