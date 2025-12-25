import cplex
from docplex.mp.model import Model
import settings # Ensure this is imported for settings.epsilon
from fractionalCover import coveredVertices
import pulp
import statistics
from fractionalCover import *


def makeBalancedSeparatorHypergraphNormalLP(H, gamma):
    # γ (gamma): function mapping the set of edges E to binary weight, 
    # defined in other parts of the code
    sum_gamma = sum(gamma[e] for e in H.E)


    # F = all edges e in eset with gamma[e] > 0
    F = {e for e in H.E if gamma[e] > 0}
    model = Model('bi_project')
   
    # x[v]: Binary, 1 if vertex v is in the separator S
    # y[e]: Continuous [0,1], 1 if edge e is covered by the separator
    x = model.binary_var_dict(H.V, name='vertex_in_sep')
    y = model.continuous_var_dict(H.E, lb=0, ub=1, name='edge_covered')
    # d[f, v]: Distance from source edge e to vertex v
    # d[f, e’]: Distance from source edge e to edge e
    d_ev = model.continuous_var_dict([(f, v) for f in F for v in H.V], lb=0, ub=1, name='Dist_f_v')
    d_ee = model.continuous_var_dict([(f, ePrime) for f in F for ePrime in F], lb=0, ub=1, name='Dist_f_e')
    # Model
    # Minimize fractional width (sum of y)
    model.minimize(model.sum(y[e] for e in H.E))
    model.add_constraints(x[v] <= model.sum(y[e] for e in H.edgesOf[v]) for v in H.V)
    Z = coveredVertices(H, gamma)
    
    # Make Gaifman Graph
    G = H.gaifmanGraph()

    if len(H.inducedSubhypergraph(H.V - Z).connectedComponents()) == 1 and all(len(H.G[v] - Z) > 0 for v in Z):
        model.add_constraint(model.sum(x[v] for v in H.V - Z) >= 1)

    # 3 Distance Constraints
    for f in F:    
        model.add_constraints(d_ev[(f,v)] <= x[v] for v in H.verticesOf[f])  #A
        for v in H.V:
            model.add_constraints(d_ev[(f,v)] <= d_ev[(f,u)] + x[v] for u in G[v])#B
        model.add_constraints(d_ee[(f, ePrime)] <= d_ev[(f,v)] for ePrime in F for v in H.verticesOf[ePrime]) #C

    for f in F:
        model.add_constraint(
            model.sum(gamma[ePrime] * d_ee[(f,ePrime)] for ePrime in F) >= (sum_gamma / 2.0) - settings.epsilon)
        
    model.print_information()
    sol = model.solve(log_output=False) 
   
    if sol:
        # Extract the separator: vertices where x[v] is roughly 1
        separator_vertices = set()
        for v in H.V:
            if x[v].solution_value == 1: # If x is 1.0 (binary)
                separator_vertices.add(v)
       
        # Return the separator set and the width (objective value)
        return separator_vertices, model.objective_value
    else:
        print("CPLEX failed to find a solution")
        return set(), 0.0

def makeBalancedSeparatorGaifmanNormalLP(H, gamma):
    if H.G is None: H.G = H.gaifmanGraph()

    sum_gamma = sum(gamma[e] for e in H.E)

    # Important edges: F = all edges e in eset with gamma[e] > 0
    F = {e for e in H.E if gamma[e] > 0} 
    
    prob = pulp.LpProblem("Balanced_Clique_Covered_Separator", pulp.LpMinimize)

    # Variables: y_e (one per edge e in E), x_v (one per vertex v in V), d_e,v (one per e in F), d_e,e' (one per e in F and e' in E)
    Y = pulp.LpVariable.dicts("y", list(H.E), 0, 1)
    
    #vertex variables are binary if solving ILP. For small instances only!
    if settings.useSeparatorLP == "GaifmanNormalILP":
        X = pulp.LpVariable.dicts("x", list(H.V), 0, None, cat="Binary")
    else:
        X = pulp.LpVariable.dicts("x", list(H.V), 0, None)

    D_ev = pulp.LpVariable.dicts("DFV", [(e,v) for e in F for v in H.V], 0, 1)
    D_ee = pulp.LpVariable.dicts("DFF", [(e1,e2) for e1 in F for e2 in F],0, 1)
    
    # Minimize sum of y_e's
    prob += (pulp.lpSum([Y[e] for e in H.E]), "Total_Covering_Cost")
    
    # x_v == sum of y_e over all e such that v is in e. 
    for v in H.V: 
        if settings.useSeparatorLP == "GaifmanNormalILP":
            prob += pulp.lpSum([Y[e] for e in H.edgesOf[v]]) >= X[v]
        else:
            prob += pulp.lpSum([Y[e] for e in H.edgesOf[v]]) == X[v]

    #must contain at least one vertex not in the set to be split.
    #only true if G-Z is connected and every vertex in Z has a neighbor outside.
    Z = coveredVertices(H, gamma)
    if len(H.inducedSubhypergraph(H.V - Z).connectedComponents()) == 1 and all(len(H.G[v] - Z) > 0 for v in Z):
        prob += pulp.lpSum([X[v] for v in H.V - Z]) >= 1

    
    #from edge to vertex in edge.
    for f in F:
        for v in H.verticesOf[f]:
            prob += D_ev[(f, v)] <= X[v]
    
    #from edge to vertex, triangle inequality:
    for f in F:
        for v in H.V:
            for u in H.G[v]:
                prob += D_ev[(f,u)] <= D_ev[(f, v)] + X[u]
    
    #from edge to edge, min vertex in edge
    for f1 in F:
        for f2 in F:
            for v in H.verticesOf[f2]:
                prob += D_ee[(f1, f2)] <= D_ev[(f1, v)]
    
    #from edge to edge, symmetric
    for f1 in F:
        for f2 in F:
            prob += D_ee[(f1, f2)] == D_ee[(f2, f1)]

    # for every edge e in F: sum over e' in F d_e,e' * gamma_e' >= Gamma / 2
    for f in F: 
        prob += pulp.lpSum([gamma[e] * D_ee[(f,e)] for e in F]) >= (sum_gamma / 2) - settings.epsilon
    
    return prob, X, Y


# Make the Balanced Separator LP (THE BOOSTED GAIFMAN VERSION)!! 
#
# the boosting only helps when X is fractional, so no point in having an ILP version of this. 
#
# H is the relevant hypergraph.
# gamma is a dict : H.E --> [0,1]. Represents a fractional cover for the vertex set Z to be separated. 
#
def makeBalancedSeparatorGaifmanBoostedLP(H, gamma):
    if H.G is None: H.G = H.gaifmanGraph()

    sum_gamma = sum(gamma[e] for e in H.E)

    # Important edges: F = all edges e in eset with gamma[e] > 0
    F = {e for e in H.E if gamma[e] > 0} 
    
    prob = pulp.LpProblem("Balanced_Clique_Covered_Separator", pulp.LpMinimize)

    # Variables: y_e (one per edge e in E), x_v (one per vertex v in V), d_e,v (one per e in F), d_e,e' (one per e in F and e' in E)
    Y = pulp.LpVariable.dicts("y", list(H.E), 0, 1)
    X = pulp.LpVariable.dicts("x", list((u, v) for u in H.V for v in H.G[u] | {u}), 0, None)
    D_ev = pulp.LpVariable.dicts("DFV", [(e,v) for e in F for v in H.V], 0, 1)
    D_ee = pulp.LpVariable.dicts("DFF", [(e1,e2) for e1 in F for e2 in F],0, 1)
    
    # Minimize sum of y_e's
    prob += (pulp.lpSum([Y[e] for e in H.E]), "Total_Covering_Cost")
    
    #x_(v,v) == sum of y_e over all e such that v is in e. 
    for v in H.V: 
        prob += pulp.lpSum([Y[e] for e in H.edgesOf[v]]) == X[(v, v)]
        
    #must contain at least one vertex not in the set to be split.
    #only true if G-Z is connected and every vertex in Z has a neighbor outside.
    Z = coveredVertices(H, gamma)
    if len(H.inducedSubhypergraph(H.V - Z).connectedComponents()) == 1 and all(len(H.G[v] - Z) > 0 for v in Z):
        prob += pulp.lpSum([X[(v, v)] for v in H.V - Z]) >= 1
        
    #x_(u,v) == sum of y_e over all e such that u is not in e and v is in e
    for v in H.V: 
        for u in H.G[v]:
            prob += pulp.lpSum( [Y[e] for e in H.edgesOf[v] if e not in H.edgesOf[u] ]) == X[(u, v)]
    
    #from edge to vertex in edge.
    for f in F:
        for v in H.verticesOf[f]:
            prob += D_ev[(f, v)] <= X[(v, v)]
    
    #from edge to vertex, triangle inequality:
    for f in F:
        for v in H.V:
            for u in H.G[v]:
                prob += D_ev[(f,u)] <= D_ev[(f, v)] + X[(v, u)]
    
    #from edge to edge, min vertex in edge
    for f1 in F:
        for f2 in F:
            for v in H.verticesOf[f2]:
                prob += D_ee[(f1, f2)] <= D_ev[(f1, v)]
    
    #check that this is ok
    #from edge to edge, symmetric
    for f1 in F:
        for f2 in F:
            prob += D_ee[(f1, f2)] == D_ee[(f2, f1)]

    # for every edge e in F: sum over e' in F d_e,e' * gamma_e' >= Gamma / 2
    for f in F: 
        #prob += pulp.lpSum([gamma[e] * D_ee[(f,e)] for e in F]) >= (sum_gamma / 2) - settings.epsilon
        #is the epsilon creating problems??
        prob += pulp.lpSum([gamma[e] * D_ee[(f,e)] for e in F]) >= (sum_gamma / 2)
    
    return prob, X, Y

def fractionalBalancedSeparator(H, gamma):  
    
    statistics.totalTimeSolvingBalSepLP -= time.time()

    if settings.useSeparatorLP == "gammaPlusFractionalCoverHeuristic":
        # 1/2gamma is a feasible solution for the LP. Add uniform-ish spread over all the vertices using fractional cover of all vertices. 
        fCovV, coverCoeff = fractionalCover(H, H.V)
        sumGamma = sum(gamma[e] for e in H.E)
        
        #scale cover coefficients so they add to sumGamma * settings.heuristicFCovWeight
        coverCoeff = {e : coverCoeff[e] * settings.heuristicFCovWeight * sumGamma / fCovV for e in H.E}
        
        # 0.5*gamma[e] is a feasible solution. Add 0.5 * coverCoeff to get uniform(ish) spread on all other vertices. 
        eVal = {e : 0.5 * (gamma[e] + coverCoeff[e]) for e in H.E}
        X_proxy = {v : sum(eVal[e] for e in H.edgesOf[v]) for v in H.V}
        vVal = {(u, v) : X_proxy[v] for v in H.V for u in (H.G[v] | {v})}

        # this LP feasilble solution can not be used as a lower bound on the fhtw, hence 0 for lpVal
        lpVal = 0
    else:
        if settings.useSeparatorLP in {"GaifmanNormal", "GaifmanNormalILP"}:
            prob, X, Y = makeBalancedSeparatorGaifmanNormalLP(H, gamma)
            prob.solve(settings.getSolver())    
            vVal = {(u, v) : X[v].varValue for v in H.V for u in (H.G[v] | {v})}    
        elif settings.useSeparatorLP == "HypergraphNormal":
            prob, X, Y = makeBalancedSeparatorHypergraphNormalLP(H, gamma)
            prob.solve(settings.getSolver())
            vVal = {(u, v) : X[v].varValue for v in H.V for u in (H.G[v] | {v})}
        elif settings.useSeparatorLP == "GaifmanBoosted":
            prob, XX, Y = makeBalancedSeparatorGaifmanBoostedLP(H, gamma)
            prob.solve(settings.getSolver())
            vVal = {(u, v) : XX[(u, v)].varValue for u in H.V for v in (H.G[u] | {u})}
            
        eVal = {e : Y[e].varValue for e in H.E}
        lpVal = sum(eVal[e] for e in H.E)

    statistics.totalTimeSolvingBalSepLP += time.time()
    
    return vVal, eVal, lpVal

