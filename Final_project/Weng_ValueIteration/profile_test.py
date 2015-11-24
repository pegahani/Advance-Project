import pylab as plt
from ValueIteration import *



n=11
x = np.r_[0:1:n*1j]

def draw_line(list_inequality_ABVI, list_inequality_IVI, lambda_rand):

    fig = plt.figure()
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)

    ax1.set_xlim([-0.5, 1.5])
    ax1.set_ylim([-0.5, 1.5])

    ax2.set_xlim([-0.5, 1.5])
    ax2.set_ylim([-0.5, 1.5])

    for l in list_inequality_ABVI:

        if l[2]!= 0:
            m = -1*l[1]/l[2]
            b = -1*l[0]/l[2]
            ax1.plot(x, m*x + b)
        else:
            ax1.plot((-l[0]/l[1], -l[0]/l[1]), (0, 1), 'k-')
        ax1.plot(lambda_rand[0], lambda_rand[1],'g^')


    for t in list_inequality_IVI:

        if t[2] != 0:
            m = -1*t[1]/t[2]
            b = -1*t[0]/t[2]
            ax2.plot(x, m*x + b)
        else:
            ax2.plot((-t[0]/t[1], -t[0]/t[1]), (0, 1), 'k-')
        ax2.plot(lambda_rand[0], lambda_rand[1],'g^')

    plt.grid()
    plt.show()

    return 0


def draw_simple_line(list_inequality):
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

    erro_average_ma = []
    query_average_ma = []

    erro_average_weng = []
    query_average_weng = []

    _state, _action, _d = 5, 2, 2

    for iteration in range(10):

        _Lambda_inequalities = generate_inequalities(_d)
        _lambda_rand = interior_easy_points(_d)

        #_r = my_mdp.generate_random_reward_function(_state, _action, _d)
        #m = my_mdp.make_simulate_mdp(_state, _action, _lambda_rand,None)
        m = my_mdp.make_simulate_mdp_Yann(_state, _action, _lambda_rand, None)

        w = Weng(m, _lambda_rand, _Lambda_inequalities)
        w.setStateAction()

        m.set_Lambda(_lambda_rand)
        Uvec = m.policy_iteration()
        exact = m.initial_states_distribution().dot(Uvec)

        # output = w.value_iteration_with_advantages(_epsilon=0.001, k=100000, noise= None,
        #                                            cluster_error = 0.01, threshold = 0.0001)

        output = w.value_iteration_with_advantages(_epsilon=0.001, k=100000, noise= 0.5,
                                                   cluster_error = 0.01, threshold = 0.0001)

        _Lambda_inequalities = generate_inequalities(_d)
        w.reset(m, _lambda_rand, _Lambda_inequalities)
        #output_weng = w.value_iteration_weng(k=100000, noise=None, threshold=0.0001)
        output_weng = w.value_iteration_weng(k=100000, noise=0.5, threshold=0.0001)


        print '***************************'
        print "_lambda_rand", _lambda_rand

        print output[1]
        print 'vector length', len(output[1])

        print output_weng[1]
        print 'vector weng length', len(output_weng[1])

        #draw_line(output[1], output_weng[1])
        #draw_simple_line(output_weng[1])

        print "***********************"
        print 'exact', exact
        print "vector list abvi", output[0]
        print "vector list weng", output_weng[0]

        graph_weng = []
        for i in output_weng[0]:
                differenece = linfDistance([np.array(i)], [np.array(exact)], 'chebyshev')[0,0]
                graph_weng.append(differenece)

        graph_ma = []
        for i in output[0]:
                differenece = linfDistance([np.array(i)], [np.array(exact)], 'chebyshev')[0,0]
                graph_ma.append(differenece)


        print 'graph ma', graph_ma
        print 'len(graph ma)', len(graph_ma)
        print 'graph_weng', graph_weng
        print 'len(graph_weng)', len(graph_weng)

        erro_average_ma.append(graph_ma)
        query_average_ma.append(range(1,len(graph_ma)+1))
        erro_average_weng.append(graph_weng)
        query_average_weng.append(range(1,len(graph_weng)+1))


        draw_line(output[1], output_weng[1], _lambda_rand)

        plt.plot(range(1,len(graph_ma)+1), graph_ma, 'r')
        plt.plot(range(1,len(graph_weng)+1), graph_weng, 'b')
        plt.show()

    print 'erro_average_ma', erro_average_ma
    print 'query_average_ma', query_average_ma

    print 'erro_average_weng', erro_average_weng
    print 'query_average_weng', query_average_weng
