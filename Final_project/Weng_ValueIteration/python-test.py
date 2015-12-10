import cplex
import numpy as np
from pulp import pulp
from scipy.spatial.distance import cdist as linfDistance

_V_best = [0.80000001, 0.]
Q = [0.,  0.2]


def pulp_K_dominance_check(_V_best, Q):
        #ineq = self.Lambda_inequalities
        ineq = [[0, 1, 0], [1, -1, 0], [0, 0, 1], [1, 0, -1]]
        _d = len(_V_best)

        prob = pulp.LpProblem("Ldominance", pulp.LpMinimize)
        lambda_variables = pulp.LpVariable.dicts("l", range(_d),lowBound=0.0, upBound=1.0)

        for inequ in ineq:
            prob += pulp.lpSum([inequ[j + 1] * lambda_variables[j] for j in range(0, _d)]) + inequ[0] >= 0

        prob += pulp.lpSum([lambda_variables[i] * (_V_best[i]-Q[i]) for i in range(_d)])

        prob.writeLP("show-pulp.lp")

        status = prob.solve()
        #LpStatus[status]

        result = pulp.value(prob.objective)

        print "result pulp", result

        if result < 0.0:
            return False

        return True

def cplex_K_dominance_check(_V_best, Q):

    #ineqList = self.Lambda_inequalities
    ineqList = [[0, 1, 0], [1, -1, 0], [0, 0, 1], [1, 0, -1]]
    _d = len(_V_best)


    prob = cplex.Cplex()
    prob.objective.set_sense(prob.objective.sense.minimize)

    c = [ float(_V_best[j]-Q[j]) for j in range(0, _d)]

    prob.variables.add(obj=c,lb = [0.0]*_d,ub = [1.0]*_d)

    prob.set_results_stream(None)
    prob.set_log_stream(None)

    constr, rhs = [], []

    for inequ in ineqList:
        c = [  [j,1.0*inequ[j + 1]] for j in range(0, _d)]
        constr.append( zip(*c) )
        rhs.append(-inequ[0])

    prob.linear_constraints.add(lin_expr=constr, senses = "G"*len(constr),rhs = rhs)
    prob.solve()

    prob.write(filename='show-cplex.lp')

    status = prob.solution.get_status_string(prob.solution.get_status())

    result = prob.solution.get_objective_value()

    print "result cplex", result

    assert status in ["infeasible","optimal"]," attention, cplex renvoie le code inconnu: "+status

    return status == "infeasible"




print '**********'

print 'PULP', pulp_K_dominance_check(_V_best, Q)
print 'CPLEX', cplex_K_dominance_check(_V_best, Q)