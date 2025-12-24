import settings
from hypergraph import *

import pulp
import time

from hypergraph import *
from fractionalCover import *

import statistics

# makes fractional cover LP for covering vertices in H.V using edges in H.E.
def makeFractionalCoverLP(H):
    prob = pulp.LpProblem("Fractional_Edge_Cover", pulp.LpMinimize)
    Y = pulp.LpVariable.dicts("y", list(H.E), 0 , 1) 

    prob += (pulp.lpSum([Y[e] for e in H.E]), "Total_Covering_Cost")
    
    for v in H.V:
        prob += (pulp.lpSum([Y[e] for e in H.edgesOf[v]]) >= 1.0, "Vertex_" + str(v))

    return prob, Y

# computes the fractional cover of H.v using H.e
def fractionalCover(H, vset):
    prob, Y = makeFractionalCoverLP(H.inducedSubhypergraph(vset))
    prob.solve(settings.getSolver())
    eset = H.incidentHyperedges(vset)
    return sum(Y[e].varValue for e in eset), {e : Y[e].varValue if e in eset else 0 for e in H.E}

#gamma is dict : H.E --> [0,1], function returns the set of vertices covered by gamma. 
def coveredVertices(H, gamma):
    coverVal = getCoverVal(H, gamma)
    return set(v for v in H.V if coverVal[v] > 1 - settings.epsilon)

def getCoverVal(H, gamma):
    coverVal = {v : 0 for v in H.V}
    for e in H.E:
        for v in H.verticesOf[e]:
            coverVal[v] += gamma[e]
    return coverVal
    
    

def makeFractionalPackingLP(H):
    prob = pulp.LpProblem("Fractional_Vertex_Packing", pulp.LpMaximize)
    X = pulp.LpVariable.dicts("x", list(H.V), 0 , 1) 

    prob += (pulp.lpSum([X[v] for v in H.V]), "Total Packing")

    for e in H.E:
        prob += (pulp.lpSum([X[v] for v in H.verticesOf[e]]) <= 1.0, "Edge_" + str(e))

    return prob, X

# computes the fractional packing of H.V with respect to H.E 
def fractionalPacking(H, vset):
    prob, X = makeFractionalPackingLP(H.inducedSubhypergraph(vset))
    prob.solve(settings.getSolver())
    return sum(X[v].varValue for v in vset), {v : X[v].varValue if v in vset else 0 for v in H.V}

# needs testing.     
def tightEdges(H, X):
    packValue = {e : 0 for e in H.E}
    for v in H.V:
        for e in H.edgesOf[v]:
            packValue[e] += X[v]
    return {e for e in H.E if packValue[e] > 1 - settings.epsilon}

#first tries to add vertices that are covered for free. If that succeeds, return. 
#If it fails try adding one at a time.
def increaseToMaxWithSameCover(H, S):
    fCovS, coverCoeff = fractionalCover(H, S)
    ans = coveredVertices(H, coverCoeff)
    
    if ans != S: return ans
    
    _, packingCoeff = fractionalPacking(H, S)
    
    candidates = H.incidentVertices(tightEdges(H, packingCoeff)) - S
    
    assert len(candidates - H.openNeighborhood(S)) == 0 , "candidate set contains elements outside open neighborhood of S."
    
    for v in candidates:
        if fractionalCover(H, (ans | {v}))[0] <= fCovS + settings.epsilon:
            ans.add(v)
    return ans
    
def greedyIntegralCover(H, gamma):
    newGamma = {e : 1 if gamma[e] > 1 - settings.epsilon else 0 for e in H.E}
    curCov = coveredVertices(H, newGamma)
    while curCov != H.V:
        bestEdge = None
        for e in H.E:
            if bestEdge == None or (len(H.verticesOf[e] & (H.V - curCov)) > len(H.verticesOf[bestEdge] & (H.V - curCov))):
                bestEdge = e
        newGamma[bestEdge] = 1
        curCov = curCov | H.verticesOf[bestEdge]
    return len([e for e in H.E if newGamma[e] >= 1-settings.epsilon]), newGamma
        