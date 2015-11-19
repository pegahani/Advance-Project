import pylab as plt
from ValueIteration import *


n=11
x = np.r_[0:1:n*1j]

def draw_line(list_inequality):

    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.set_xlim([-0.5, 1.5])
    ax1.set_ylim([-0.5, 1.5])

    for l in list_inequality:

        if l[2] != 0:
            m = -1*l[1]/l[2]
            b = -1*l[0]/l[2]
            ax1.plot(x, m*x + b)
        else:
            ax1.plot((-l[0]/l[1], -l[0]/l[1]), (0, 1), 'k-')

    plt.grid()
    plt.show()

    return 0


if __name__ == '__main__':

    _state, _action, _d = 3, 2, 2

    _Lambda_inequalities = generate_inequalities(_d)
    _lambda_rand = interior_easy_points(_d)

    print "_lambda_rand", _lambda_rand

    #_r = my_mdp.generate_random_reward_function(_state, _action, _d)
    #m = my_mdp.make_simulate_mdp(_state, _action, _lambda_rand,None)
    m = my_mdp.make_simulate_mdp_Yann(_state, _action, _lambda_rand,None)

    w = Weng(m, _lambda_rand, _Lambda_inequalities)
    w.setStateAction()

    output = w.value_iteration_with_advantages(_epsilon=0.001, k=100000, noise= None,
                                               cluster_error = 0.01, threshold = 0.0001)


    # _Lambda_inequalities = generate_inequalities(_d)
    # w.reset(m, _lambda_rand, _Lambda_inequalities)
    # output_weng = w.value_iteration_weng(k=100000, noise=None, threshold=0.0001)


    print output[1]
    print 'vector length', len(output[1])
    draw_line(output[1])

    # print output_weng[1]
    # print 'vector weng length', len(output_weng[1])
    #draw_line(output_weng[1])
