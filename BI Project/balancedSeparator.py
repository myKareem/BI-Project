import sys
import random
import time

import settings
import statistics
import turboShrinkComponent

from biProjectCode import *
from hypergraph import *
from fractionalCover import *

import cplex
from docplex.mp.model import Model
import settings

def makeBalancedSeparatorHypergraphNormalLP(H, gamma):
    sum_gamma = sum(gamma[e] for e in H.E)

    # Important edges: F = all edges e in eset with gamma[e] > 0
    F = {e for e in H.E if gamma[e] > 0} 
    
    model = Model('bi_project')
    
    # x[v]: Binary, 1 if vertex v is in the separator S
    # y[e]: Continuous [0,1], 1 if edge e is covered by the separator
    x = model.binary_var_dict(H.V, name='vertex_in_sep')
    y = model.continuous_var_dict(H.E, lb=0, ub=1, name='edge_covered')
    
    # d[f, v]: Distance from source edge e to vertex v
    # d[f, e]: Distance from source edge e to edge e
    d_ev = model.continuous_var_dict([(f, v) for f in F for v in H.V], lb=0, ub=1, name='Dist_f_v')
    d_ee = model.continuous_var_dict([(f, ePrime) for f in F for ePrime in F], lb=0, ub=1, name='Dist_f_e')
    
    # Model
    # Minimize fractional width (sum of y)
    model.minimize(model.sum(y[e] for e in H.E))
    
    # 1 Cover Constraint
    # FIX: Used plural 'add_constraints' correctly for generator
    model.add_constraints(x[v] <= model.sum(y[e] for e in H.edgesOf[v]) for v in H.V)
    
    # FIX: Ensure 'coveredVertices' and 'inducedSubhypergraph' are available. 
    # If they are from the original 'hypergraph.py', ensure 'H' has these methods.
    Z = coveredVertices(H, gamma)
    
    # FIX: 
    # 1. Used 'model.add_constraint' instead of 'prob +=' (PuLP syntax).
    # 2. Used 'model.sum' instead of 'pulp.lpSum'.
    # 3. Used 'x' (lowercase) instead of 'X' (undefined).
    if len(H.inducedSubhypergraph(H.V - Z).connectedComponents()) == 1 and all(len(H.G[v] - Z) > 0 for v in Z):
        model.add_constraint(model.sum(x[v] for v in H.V - Z) >= 1)
    
    # 2 Distance Constraints
    # FIX: H.gaifmanGraph() usually returns an adjacency dict {v: {neighbors}}.
    G = H.gaifmanGraph() 
    
    for f in F:    
        # FIX: 'add_constraints' (plural) because you are passing a generator (for v in ...)
        model.add_constraints(d_ev[(f,v)] <= x[v] for v in H.verticesOf[f])
        
        # FIX: Typo 'add_conatraint' -> 'add_constraints'
        # Note: This is redundant if variable ub=1, but syntactically correct now.
        model.add_constraints(d_ev[(f,v)] <= 1 for v in H.verticesOf[f]) 
        
        # FIX: Triangle Inequality Logic
        # 1. 'u' is a vertex, so you cannot use d_ee[(f,u)]. Changed to d_ev[(f,u)].
        # 2. Iterating 'for v in G for u in G' creates O(|V|^2) constraints (all pairs).
        #    You likely meant neighbors: 'for u in G[v]'.
        #    I kept your loop structure but fixed the indexing error.
        for v in H.V:
            # Assuming G is an adjacency dict {v: set(neighbors)}
            model.add_constraints(d_ev[(f,v)] <= d_ev[(f,u)] + x[v] for u in G[v])
            
        # FIX: ePrime loop
        model.add_constraints(d_ee[(f, ePrime)] <= d_ev[(f,v)] for ePrime in F for v in H.verticesOf[ePrime])
    
    # 3 Balance Constraint
    # FIX: The original constraint applies PER source edge 'f'.
    # Your code tried to sum over 'f' inside the constraint, making one giant constraint.
    # I moved the 'for f in F' loop outside.
    for f in F:
        model.add_constraint(
            model.sum(gamma[ePrime] * d_ee[(f,ePrime)] for ePrime in F) >= (sum_gamma / 2.0) - settings.epsilon
        )
    
    model.print_information()
    #sol = model.solve(log_output=True)
    
    # FIX: Access solve details from the solution object or model, ensuring safe access
    #if sol:
        #print("Solution found")
        # print(sol.get_objective_value())
    #else:
        #print("No solution")
        
    #return model
    sol = model.solve(log_output=False) # Turn off log to keep terminal clean
    
    if sol:
        # Extract the separator: vertices where x[v] is roughly 1
        separator_vertices = set()
        for v in H.V:
            if x[v].solution_value > 0.9: # If x is 1.0 (binary)
                separator_vertices.add(v)
        
        # Return the separator set and the width (objective value)
        return separator_vertices, model.objective_value
    else:
        print("CPLEX failed to find a solution")
        return set(), 0.0

# Round LP
#
# gamma is the dict vset --> [0,1] to separate.
# X is dict V x V --> [0,1] = cost to pass through an edge. 
# Y is dict E --> [0,1] = coverage coefficient of edge. 
#     
# while gamma-heaviest component C of G-S has > 1/2 of gamma: 
#       grow S to shrink C.

def roundSeparatorLP(H, gamma, X, Y, *args):
    statistics.totalTimeRoundingBalSepLP -= time.time()

    sumGamma = sum(gamma[e] for e in H.E)
    
    if len(args) > 0:
        careAboutRootCover = args[0]
    else:
        careAboutRootCover = False

    Z = coveredVertices(H, gamma)
    assert len(Z - H.V) == 0, "Z contains vertices outside H.V!"
    S = set()
    C = set(H.V)
    largeComponentZCover = sum(gamma[e] for e in H.E)
    largeComponentNewBagCover = float('inf')
   
    roundingLoop = 0
    while largeComponentZCover * 2 > sumGamma and (not careAboutRootCover or largeComponentNewBagCover >= sumGamma):
        S, C, largeComponentZCover = shrinkLargeComponent(H.inducedSubhypergraph(S | C | Z), gamma, X, Y, S, C, Z, careAboutRootCover)

        if settings.useZCoverEstimator: largeComponentZCover, _ = fractionalCover(H, Z & C)
        largeComponentNewBagCover, _ = fractionalCover(H, (Z & C) | H.openNeighborhood(C))
        
        assert len(S - H.V) == 0, "S contains vertices outside H.V!"
        assert len(C - H.V) == 0, "C contains vertices outside H.V!"
        assert len(Z - H.V) == 0, "Z contains vertices outside H.V!"

        roundingLoop += 1
    
    statistics.maxRoundingLoops = max(statistics.maxRoundingLoops, roundingLoop)
    statistics.totalTimeRoundingBalSepLP += time.time()
    return S
    
    
#computes a list of candidate separator sets to try to remove from current C_hat.
#one candidate is the y-heaviest edge. The others are "X-BFS"-layers from the gamma-heaviest edge. 
def computeCandidateSets(H, gamma, X, Y):
    heaviestYEdge = largestElementOf(H.E, Y)
    ans = [H.verticesOf[heaviestYEdge]]
    
    assert len(H.verticesOf[heaviestYEdge]) > 0, "Heaviest Y-edge is empty!"

    # do the add remove sequence from the heaviest gamma edge. Can try other options here!!
    if settings.candidateSetRoot == "heaviestGammaEdge":
        heaviestGammaEdge = largestElementOf(H.E, gamma)
        ans.extend(computeCandidateSetsFromEdge(H, X, heaviestGammaEdge))
    elif settings.candidateSetRoot == "allNonzeroGammaEdge":
        for e in H.E:
            if gamma[e] < settings.epsilon: continue
            ans.extend(computeCandidateSetsFromEdge(H, X, e))
    elif settings.candidateSetRoot == "randomNonzeroGammaEdge":
        randomGammaEdge = selectRandomElementOutside(H.E, set(), gamma)
        ans.extend(computeCandidateSetsFromEdge(H, X, randomGammaEdge))
    
    return ans
    

def computeCandidateSetsFromEdge(H, X, sourceEdge):
    ans = []
    vertexOrdering = addRemoveSequence(H, X, sourceEdge)
    
    #print("Add remove sequence: ", end="")
    #self.printVertexSet(vertexOrdering)
    
    previousD = 0
    candS = set()
    for (d, v) in vertexOrdering:
        if len(candS) > 0 and (not settings.pruneCandidateSets or d > previousD + settings.epsilon):
            ans.append(set(candS))   
            previousD = d
        if v in candS: candS.remove(v); 
        else: candS.add(v)
    
    if len(candS) > 0 and (not settings.pruneCandidateSets or d > previousD + settings.epsilon): ans.append(set(candS))   
    return ans

    
        
#V is vertex set, X is X-values and D are D-values per vertex. Sort the endpoints of all intervals [D[v]-X[v], D[v]]
def addRemoveSequence(H, X, startEdge):
    endD = H.dijkstra(X, H.verticesOf[startEdge])
    startD = {v : 0 if v in H.verticesOf[startEdge] else min(endD[u] for u in H.G[v]) - (settings.epsilon*settings.epsilon) for v in H.V}
    pairList = [(endD[v], v) for v in H.V if X[(v,v)] > settings.epsilon] + [(startD[v], v) for v in H.V if X[(v,v)] > settings.epsilon]
    pairList.sort()
    return pairList


#returns the first element of container whose weight is largest. 
def largestElementOf(container, weight):
    ans = None
    for elt in container: 
        if ans is None or weight[elt] > weight[ans]: ans = elt
    return ans



def newLossGainPairIsBetter(newLoss, newGain, oldLoss, oldGain):
    if oldLoss == None or newLoss == None:
        return True
    
    if newLoss < 0:
        return newLoss < oldLoss
    elif oldLoss < 0:
        return False
    
    return newLoss * oldGain < oldLoss * newGain


# for each candidate set S: 
#    compute gamma-heaviest component C of G-(S union S_hat). C is empty if none > gamma/2. 
#    update S to N(C) if C is non-empty
#    loss = fcov(Z cup S) - fcov(Z cup S_hat)
#    gain = decrease in y_e's in C_hat to C. 
# Return S with minimum loss/gain (if loss negative then stop and keep) (if gain is 0 then skip)  
def shrinkLargeComponent(H, gamma, X, Y, S, C, Z, careAboutRootCover):

    #it is important that the sum here is taken over gamma and not H.E since H changes but gamma does not,
    sumGamma = sum(gamma[e] for e in gamma)
    coverVal = getCoverVal(H, gamma)

    assert len(C - H.V) == 0, "C contains vertices outside H!"
    assert len(C) > 0, "C is empty"
    
    if settings.turboRounding:
        heaviestYEdge = largestElementOf(H.incidentHyperedges(C), Y)
        candidateSets = [H.verticesOf[heaviestYEdge] & C]

        #add support for settings selecting type of edge to round from in turbo mode.
        assert settings.candidateSetRoot == "randomNonzeroGammaEdge"
        randomGammaEdge = selectRandomElementOutside(H.incidentHyperedges(C), set(), gamma)
        bestCandidateSet, claimedGain, claimedLoss = turboShrinkComponent.bestCandidateSetFromEdge(H, gamma, X, Y, S, C, Z, randomGammaEdge, careAboutRootCover)
        candidateSets.append( bestCandidateSet )
    else:
        candidateSets = computeCandidateSets(H.inducedSubhypergraph(C), gamma, X, Y)
    
    statistics.maxCandidateSets = max(statistics.maxCandidateSets, len(candidateSets))
    
    assert len(candidateSets) > 0, "No candidate sets!"
    assert all(len(candS) > 0 for candS in candidateSets), "Empty candidate set!"
    
    #Will Z be added to the separator or not? This affects whether we include fcov of Z in the loss. 
    if careAboutRootCover:
        baseLoss, _ = fractionalCover(H, (Z | S))
        baseLossEstimator = sum(Y[e] * max((1-coverVal[v])/X[(v,v)] for v in H.verticesOf[e] if v in S) for e in H.incidentHyperedges(S) if Y[e] > 100 * settings.epsilon)
    else:
        baseLoss, _ = fractionalCover(H, S)
        baseLossEstimator = sum(Y[e] * max(1/X[(v,v)] for v in H.verticesOf[e] if v in S) for e in H.incidentHyperedges(S) if Y[e] > 100 * settings.epsilon)       

    sumY = sum(Y[e] for e in H.incidentHyperedges(C))

    bestGain, bestLoss, bestSeparator, bestComponent, bestComponentZCover = None, None, None, None, None
    
    for i in range(len(candidateSets)):
        candS = candidateSets[i]
        assert len(candS - C) == 0, "candS contains vertices outside of C!"
        
        possibleHeavyComponent, possibleHeavyComponentZCover = findHeaviestComponent(H.inducedSubhypergraph(C - candS), gamma)

        candExtendedS, newHeavyComponent, newComponentZCover = inferredSeparatorAndHeavyComponent(H, sumGamma, S, C, Z, candS, possibleHeavyComponent, possibleHeavyComponentZCover)
        
        #calculate loss. Loss can be negative (in that case great). 
        loss = calculateLoss(H, X, Y, Z, coverVal, baseLoss, baseLossEstimator, candExtendedS, careAboutRootCover)
        
        # Calculate gain. Gain is non-negative, can be 0.
        gain = calculateGain(H,Y,sumY,newHeavyComponent)
        
        #update gain/loss thing to fix this.
        #this seems to be set off by floating point errors relatively often, so decreasing sensitivity here by 100. 
        #if settings.turboRounding and i == 1:
        #        assert abs(loss-claimedLoss) < settings.epsilon*100, "Claimed loss not equal to computed loss: (%.3f, %.3f)" % (loss, claimedLoss)
        #        assert abs(gain-claimedGain) < settings.epsilon*100, "Claimed gain not equal to computed gain: (%.3f, %.3f)" % (gain, claimedGain)

        #adjust gain to include decrease in number of vertices. 
        vertexGain = (len(C) - len(newHeavyComponent)) / len(C)
        gain = (1 - settings.vertexGainWeight) * gain + (settings.vertexGainWeight * vertexGain)
        
        if newLossGainPairIsBetter(loss, gain, bestLoss, bestGain): 
            bestGain = gain
            bestLoss = loss
            bestSeparator = set(candExtendedS)
            bestComponent = set(newHeavyComponent)
            bestComponentZCover = newComponentZCover

    return bestSeparator, bestComponent, bestComponentZCover
    

def inferredSeparatorAndHeavyComponent(H, sumGamma, S, C, Z, candS, possibleHeavyComponent, possibleHeavyComponentZCover):

    if possibleHeavyComponentZCover * 2 > sumGamma + settings.epsilon and not settings.useZCoverEstimator:
        possibleHeavyComponentZCover, _ = fractionalCover(H, Z & possibleHeavyComponent)
        
    #determine whether the heaviest component is light enough to be done (according to gamma)
    if possibleHeavyComponentZCover * 2 <= sumGamma + settings.epsilon:
        candExtendedS = S | candS
        newHeavyComponent = set() 
        newComponentZCover = 0
    else: 
        # update additive set to N(new heavy component). Can only do if there is a heavy component. 
        candExtendedS = H.openNeighborhood(possibleHeavyComponent)
        newHeavyComponent = possibleHeavyComponent
        newComponentZCover = possibleHeavyComponentZCover
    
    assert len(candExtendedS - H.V) == 0, "candExtendedS contains vertices outside H.V!"
    
    return candExtendedS, newHeavyComponent, newComponentZCover

   
#calculate loss. Loss can be negative (in that case great). 
def calculateLoss(H, X, Y, Z, coverVal, baseLoss, baseLossEstimator, candExtendedS, careAboutRootCover):

    if settings.useLossEstimator:
        if careAboutRootCover:
            loss = sum(Y[e] * max((1-coverVal[v])/X[(v,v)] for v in H.verticesOf[e] if v in candExtendedS) for e in H.incidentHyperedges(candExtendedS) if Y[e] > 100 * settings.epsilon) - baseLossEstimator
        else:
            loss = sum(Y[e] * max(1/X[(v,v)] for v in H.verticesOf[e] if v in candExtendedS) for e in H.incidentHyperedges(candExtendedS) if Y[e] > 100 * settings.epsilon) - baseLossEstimator
    else:
        if careAboutRootCover:
            loss = fractionalCover(H, Z | candExtendedS)[0] - baseLoss
        else:
            loss = fractionalCover(H, candExtendedS)[0] - baseLoss
            
    return loss

def calculateGain(H,Y,sumY,newHeavyComponent):
    gain = (sumY - sum(Y[e] for e in H.incidentHyperedges(newHeavyComponent))) / sumY
    assert gain > 100 * -settings.epsilon, "gain was " + str(gain)
    assert gain <= 1, "gain was more than 1!"
    return gain



def findHeaviestComponent(H, gamma):
    if len(H.V) == 0: return set(), 0
    candComponents = H.connectedComponents()
    candCompGamma = [sum(gamma[e] for e in H.incidentHyperedges(candComponents[i])) for i in range(len(candComponents))]
    maxComponentIndex = largestElementOf(range(len(candCompGamma)), candCompGamma)
    return candComponents[maxComponentIndex], candCompGamma[maxComponentIndex]
    

#use depth=1 to not reduce the balanced separator with respect to the root bag before returning.
def computeBalancedSeparator(H, root, depth, *args):
    # 1. Calculate Gamma (Edge Weights)
    # We keep this line because your function needs 'gamma' as input
    _, gamma = fractionalCover(H, root)

    # 2. Optional: Print progress bars
    if settings.printProgress: 
        print("| "*(depth-1) + "+-", end="--", file=sys.stderr)
        
    # -------------------------------------------------------------
    # [NEW] Call your Custom CPLEX Logic
    # -------------------------------------------------------------
    # Your function returns: (set_of_vertices, objective_value)
    balSep, optLP = makeBalancedSeparatorHypergraphNormalLP(H, gamma)
    
    # 3. Print the result for debugging (optional)
    if settings.printProgress: 
        print(f" CPLEX Found Sep: {len(balSep)} vertices, Width: {optLP}", file=sys.stderr)

    # 4. Safety Check (Preserves original logic)
    # Ensure we don't return vertices that don't exist in the graph
    assert balSep.issubset(H.V), "Balanced separator contains vertices outside H.V!"

    return balSep, optLP


def selectRandomElementOutside(container, setToAvoid, weight):
    sumOutsideX = sum(weight[v] if v not in setToAvoid else 0 for v in container)
    assert sumOutsideX > 0, "weight has no mass outside of set to avoid!"
    rValue = random.uniform(0, sumOutsideX)
    for v in container:
        if v in setToAvoid: continue
        rValue -= weight[v]
        if rValue <= 0:
            return v
    assert False, "Failed to select vertex"
    
def reduceBalSepWrtRootBag(H, S, root, gamma):
    CC = H.inducedSubhypergraph(H.V - (S|root)).connectedComponents()
    gammaOfComponent = {i : sum(gamma[e] for e in H.incidentHyperedges(CC[i])) for i in range(len(CC))}
    componentIndexSorted = sorted(range(len(CC)), key=lambda i : gammaOfComponent[i], reverse=True)
    

    sumGamma = sum(gammaOfComponent[i] for i in gammaOfComponent)
    totGamma = 0
    newS = set()
    
    for i in componentIndexSorted:
        newS.update(H.openNeighborhood(CC[i]) & S)
        totGamma += gammaOfComponent[i]
        if totGamma >= sumGamma / 2: break

    assert newS.issubset(S), "newS is not subset of S during trimming!"
    
    #if broke progress on graph, restore some.
    if len(newS - root) == 0 and len(S - root) > 0: newS.add(random.choice(list(S-root)))

    #if fractionalCover(H, root | newS)[0] < fractionalCover(H, root | S)[0]: print("REDUCED!")
    return newS    