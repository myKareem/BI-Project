from re import U
import sys
import random
from sortedcontainers import SortedSet
from collections import deque

import settings
import statistics
from biProjectCode import *
from hypergraph import *
from fractionalCover import *

class approximateDynamicFractionalCover:

    #probably do not need gamma?
    def __init__(self, H, gamma, Y, factor):
        
        nonZeroYEdges = {e for e in H.E if Y[e] > 0}

        self.H = H.subhypergraph(H.V, nonZeroYEdges)
        self.gamma = gamma
        self.Y = Y
        self.factor = factor

        self.activeVertices = set()
        self.activeVerticesOfEdge = {e : SortedSet(key=lambda x : self.factor[x] ) for e in self.H.E}
        self.coveringCost = 0

    def addVertex(self, v):
        if v in self.activeVertices: return
        self.activeVertices.add(v)
        for e in self.H.edgesOf[v]:
            costBeforeUpdate = self.currentCoverCost(e)
            self.activeVerticesOfEdge[e].add(v)
            costAfterUpdate = self.currentCoverCost(e)
            self.coveringCost += costAfterUpdate - costBeforeUpdate

        if settings.debug:
            self.checkInvariants()

    def removeVertex(self, v):
        if v not in self.activeVertices: return
        self.activeVertices.remove(v)
        for e in self.H.edgesOf[v]:
            costBeforeUpdate = self.currentCoverCost(e)
            assert v in self.activeVerticesOfEdge[e]
            self.activeVerticesOfEdge[e].remove(v)
            costAfterUpdate = self.currentCoverCost(e)
            self.coveringCost += costAfterUpdate - costBeforeUpdate
        
        if settings.debug:
            self.checkInvariants()

    def currentCoverCost(self, e):
        if len(self.activeVerticesOfEdge[e]) == 0: return 0
        return self.Y[e] * self.factor[self.activeVerticesOfEdge[e][-1]]

    def checkInvariants(self):
        for e in self.H.E:
            if len(self.H.verticesOf[e] & self.activeVertices) > 0:
                assert abs(self.currentCoverCost(e) - self.Y[e] * max(self.factor[v] for v in self.H.verticesOf[e] & self.activeVertices)) < settings.epsilon * 100
            else:
                assert len(self.activeVerticesOfEdge[e]) == 0

        assert abs(self.coveringCost - sum(self.currentCoverCost(e) for e in self.H.E)) < settings.epsilon*100

class outsideComponents:

    #looks like i can remove Z from here.
    def __init__(self, H, gamma, X, Y, S, Z, factor):
        self.H = H
        self.gamma = gamma
        
        #it is important that this sum is going over gamma and not H.E since H changes but gamma does not. 
        self.sumGamma = sum(gamma[e] for e in gamma)
        self.X = X
        self.Y = Y
        self.S = S
        self.Z = Z
        self.factor = factor
        
        self.Q = set()
        self.QEdges = set()
        self.parentID = {}
        self.componentVertices = {}
        self.componentY = {}
        self.componentGamma = {}

        self.hasLargeComponent = False
        self.largeComponentID = None
        self.largeComponentNeighbors = approximateDynamicFractionalCover(H, gamma, Y, factor)

    #merges two components. Does NOT the required updates to neighbors if one of them is or becomes the large component
    def union(self, comp1ID, comp2ID):
        comp1ID = self.find(comp1ID)
        comp2ID = self.find(comp2ID)

        #want comp1 to have the largest size.
        if len(self.componentVertices[comp2ID]) > len(self.componentVertices[comp1ID]):
            #if largeComponent is being swapped, update largeComponentID.
            comp1ID, comp2ID = comp2ID, comp1ID
            if self.largeComponentID  == comp1ID or self.largeComponentID == comp2ID: 
                self.largeComponentID = comp1ID

        self.componentY[comp1ID] += self.componentY[comp2ID]
        self.componentGamma[comp1ID] += self.componentGamma[comp2ID]

        #move all vertices from comp2 to comp1
        for u in self.componentVertices[comp2ID]:
            self.parentID[u] = comp1ID
        
        self.componentVertices[comp1ID].update(self.componentVertices[comp2ID]) 
        del self.componentVertices[comp2ID]

    #since for every component we also need to keep a list of the vertices
    #we do "naive" n log n union find instead of the smart one. 
    def find(self, id):
        return self.parentID[id]

    #adds new vertex to outside components. If relevant, updates the neighborhood of the large component. 
    def addVertex(self, v):
        
        assert v not in self.Q

        self.Q.add(v)
        self.parentID[v] = v
        self.componentVertices[v] = {v}
        self.componentGamma[v] = 0
        self.componentY[v] = 0

        for e in self.H.edgesOf[v]:
            if e not in self.QEdges:
                self.componentGamma[v] += self.gamma[e]
                self.componentY[v] += self.Y[e]
                self.QEdges.add(e)


        for u in self.H.G[v] & self.Q:
            if self.find(u) != self.find(v): 
                #if u or v is in the large component, add the neighborhood of the other component to the neighborhood of large component.
                if self.largeComponentID == self.find(u):
                    for w in self.H.openNeighborhood(self.componentVertices[self.find(v)]) - self.Q:
                        self.largeComponentNeighbors.addVertex(w)

                elif self.largeComponentID == self.find(v):
                    for w in self.H.openNeighborhood(self.componentVertices[self.find(u)]):
                        self.largeComponentNeighbors.addVertex(w)

                self.union(u, v)
        
        newCompID = self.find(v)

        self.largeComponentNeighbors.removeVertex(v)

        #if newCompID is about to become the big component, compute its neighborhood.
        if not self.hasLargeComponent and self.componentGamma[newCompID] * 2 > self.sumGamma + settings.epsilon:
            self.hasLargeComponent = True
            self.largeComponentID = newCompID
            for z in self.H.openNeighborhood(self.componentVertices[newCompID]):
                self.largeComponentNeighbors.addVertex(z)
            
        if settings.debug:
            #checks that parentID send self.componentVertices are consistent, 
            #and that every vertex in Q is in precisely one component. 
            count = {q : 0 for q in self.Q}
            for cc in self.componentVertices:
                for u in self.componentVertices[cc]:
                    count[u] += 1
            assert count == {q : 1 for q in self.Q}

            assert all(q in self.componentVertices[self.find(q)] for q in self.Q)
            assert all(self.parentID[q] == cc for cc in self.componentVertices for q in self.componentVertices[cc])


        if settings.debug and self.hasLargeComponent:
            assert self.largeComponentID == self.find(self.largeComponentID)
            assert self.H.openNeighborhood(self.componentVertices[self.largeComponentID]) == self.largeComponentNeighbors.activeVertices

            assert abs(self.largeComponentNeighbors.coveringCost - sum(self.Y[e] * max(self.factor[z] for z in self.H.verticesOf[e] & self.largeComponentNeighbors.activeVertices) for e in self.H.incidentHyperedges(self.largeComponentNeighbors.activeVertices) if self.Y[e] > 0)) < settings.epsilon*100

    def largeComponentY(self): 
        return self.componentY[self.largeComponentID]

    def largeComponentGamma(self): 
        return self.componentGamma[self.largeComponentID]


class insideComponent:
    def __init__(self, H, C, gamma, Y, factor):
        self.H = H
        self.Y = Y
        self.gamma = gamma
        self.sumGamma = sum(gamma[e] for e in gamma)
        
        self.vertices = set(C)
        
        self.neighborhood = approximateDynamicFractionalCover(H, gamma, Y, factor)
        for u in H.openNeighborhood(C):
            self.neighborhood.addVertex(u)

        self.numNeighborsInComponent = {v : len(H.G[v] & self.vertices) for v in self.neighborhood.activeVertices}
        self.intersectionWithComponent = {e : len(H.verticesOf[e] & self.vertices) for e in H.E}
        
        self.componentGamma = sum(gamma[e] for e in H.E if self.intersectionWithComponent[e] > 0)
        self.componentY = sum(Y[e] for e in H.E if self.intersectionWithComponent[e] > 0)

    def removeVertex(self, v):
        if v not in self.vertices: return

        self.vertices.remove(v)
        
        for u in self.H.G[v] & self.neighborhood.activeVertices:
            if self.numNeighborsInComponent[u] == 1:
                self.neighborhood.removeVertex(u)
            self.numNeighborsInComponent[u] -= 1

        for e in self.H.edgesOf[v]:
            if self.intersectionWithComponent[e] == 1:
                self.componentGamma -= self.gamma[e]
                self.componentY -= self.Y[e]
            self.intersectionWithComponent[e] -= 1    

        self.neighborhood.addVertex(v)
        self.numNeighborsInComponent[v] = len(self.H.G[v] & self.vertices)


    def isLargeComponent(self):
        return self.componentGamma * 2 > self.sumGamma + settings.epsilon


#duplicated from balanced separator to avoid circular import. Should be resolved somehow...
def newLossGainPairIsBetterTurbo(newLoss, newGain, oldLoss, oldGain):
    if oldLoss == None or oldGain == None:
        return True
    
    if newLoss < -settings.epsilon:
        return newLoss < oldLoss
    elif oldLoss < -settings.epsilon:
        return False
    
    newLoss = max(newLoss, 0)
    oldLoss = max(oldLoss, 0)

    return newLoss * oldGain < oldLoss * newGain



# Change this to generate one candidate set. 
def bestCandidateSetFromEdge(H, gamma, X, Y, S, C, Z, sourceEdge, careAboutRootCover):

    sumGamma = sum(gamma[e] for e in gamma)
    coverVal = getCoverVal(H, gamma)
    
    #Will Z be added to the separator or not? This affects whether we include fcov of Z in the loss. 
    # compute factor[v] = (1-gamma(v))/X[v] for every v in C union S.
    # if careAboutRootCover is false it is 1/X[v]
    if careAboutRootCover:
        factor = {v : (1-coverVal[v])/X[(v,v)] if X[(v,v)] > 0 else 0 for v in H.V}
    else:
        factor = {v : 1/X[(v,v)] if X[(v,v)] > 0 else 0 for v in H.V}
    
    baseLossEstimator = sum(Y[e] * max(factor[v] for v in H.verticesOf[e] if v in S) for e in H.incidentHyperedges(S) if Y[e] > settings.epsilon)
    sumY = sum(Y[e] for e in H.incidentHyperedges(C))

    #need more things to check :)
    assert len(C - H.V) == 0, "C contains vertices outside H!"
    assert len(C) > 0, "C is empty"
    
    bestSeparator, bestLoss, bestGain, bestQPos = None, None, None, None
    
    # Get the add remove sequence
    vertexQueue = addRemoveSequenceTurbo(H.inducedSubhypergraph(C), X, sourceEdge)

    #initially R=C, while S_hat = S, Q is are empty.
    R = insideComponent(H,C,gamma,Y,factor)

    S_hat = approximateDynamicFractionalCover(H, gamma, Y, factor)
    for v in S: S_hat.addVertex(v)

    #keep track of vertices in sHat intersection c.
    curSep = set()
    

    Q = outsideComponents(H,gamma,X,Y,S,Z,factor)




    for queuePos in range(len(vertexQueue)):
        vertexToMove = vertexQueue[queuePos]

        if vertexToMove not in S_hat.activeVertices:
            S_hat.addVertex(vertexToMove)
            R.removeVertex(vertexToMove)
            curSep.add(vertexToMove)
        else:
            S_hat.removeVertex(vertexToMove)
            Q.addVertex(vertexToMove)
            curSep.remove(vertexToMove)

        if R.isLargeComponent():
            gain = (sumY - R.componentY) / sumY
            loss = R.neighborhood.coveringCost - baseLossEstimator
        elif Q.hasLargeComponent:
            gain = (sumY - Q.largeComponentY()) / sumY
            loss = Q.largeComponentNeighbors.coveringCost - baseLossEstimator
        else:
            gain = sumY / sumY
            loss = S_hat.coveringCost - baseLossEstimator

        ############ DEBUG MODE ###############
        ## pretty slow. Turn off debug mode when benchmarking!!
        ########################################

        if settings.debug:
            #every computed component is a component of C-curSep
            componentsNaive =  H.inducedSubhypergraph(C-curSep).connectedComponents()
            for v in Q.componentVertices:
                cc = Q.componentVertices[v]
                if(len(cc) > 0):
                    assert cc in componentsNaive, H.printVertexSet(cc) + " not a connected component of C-curSep" + H.printVertexSet(C) + H.printVertexSet(curSep)
            if(len(R.vertices) > 0):
                assert R.vertices in componentsNaive, H.printVertexSet(R.vertices) + " not a connected component of C-curSep"  + H.printVertexSet(C) + H.printVertexSet(curSep)

            #every component of C-CurSep is accounted for. 
            for cc in componentsNaive:
                foundComponent = False
                if cc == R.vertices: foundComponent = True
                for v in Q.componentVertices:
                    if cc == Q.componentVertices[v]:
                        foundComponent = True
                        break
                assert foundComponent, "Could not find " + str(cc)
        
            #for every component, check that its Y and gamma values are correct.
            assert abs(R.componentGamma - sum(gamma[e] for e in H.incidentHyperedges(R.vertices))) < settings.epsilon
            assert abs(R.componentY - sum(Y[e] for e in H.incidentHyperedges(R.vertices))) < settings.epsilon
        
            for v in Q.componentVertices:
                cc = Q.componentVertices[v]
                assert abs(Q.componentGamma[v] - sum(gamma[e] for e in H.incidentHyperedges(cc))) < settings.epsilon
                assert abs(Q.componentY[v] - sum(Y[e] for e in H.incidentHyperedges(cc))) < settings.epsilon

            #can't have two large components at the same time
            assert not (R.isLargeComponent() and Q.hasLargeComponent)

            #check that a large component has been found if and only if there is one. 
            shouldBeLargeComponent = False
            inferredSeparator = set()
            for cc in componentsNaive:
                if sum(gamma[e] for e in H.incidentHyperedges(cc)) * 2 > sumGamma + settings.epsilon:
                    shouldBeLargeComponent = True
                    inferredSeparator = H.openNeighborhood(cc) 

            assert shouldBeLargeComponent == (R.isLargeComponent() or Q.hasLargeComponent)

            #check that the inferred separator, and its gain and loss is what it should be. 
            if shouldBeLargeComponent:
                if R.isLargeComponent():
                    assert inferredSeparator == R.neighborhood.activeVertices
                    assert abs(R.neighborhood.coveringCost - sum(Y[e] * max(factor[z] for z in H.verticesOf[e] & inferredSeparator) for e in H.incidentHyperedges(inferredSeparator) if Y[e] > 0)) < settings.epsilon*100
                else:
                    assert inferredSeparator == Q.largeComponentNeighbors.activeVertices
                    assert abs(Q.largeComponentNeighbors.coveringCost - sum(Y[e] * max(factor[z]for z in H.verticesOf[e] & inferredSeparator) for e in H.incidentHyperedges(inferredSeparator) if Y[e] > 0)) < settings.epsilon*100
        
        ################################################# END DEBUG #####################################################################################

        if len(curSep) > 0 and newLossGainPairIsBetterTurbo(loss, gain, bestLoss, bestGain):
            bestLoss = loss
            bestGain = gain
            bestQPos = queuePos

        #print(queuePos, ":", H.printVertexSet(S_hat.activeVertices), gain, loss, bestQPos)

    #print("bestQPos:", bestQPos)
    bestSeparator = set()
    for i in range(bestQPos + 1):
        v = vertexQueue[i]
        if v not in bestSeparator:
            bestSeparator.add(v)
        else:
            bestSeparator.remove(v)
    #print ("bestsep", H.printVertexSet(bestSeparator))

    return bestSeparator, bestGain, bestLoss


#V is vertex set, X is X-values and D are D-values per vertex. Sort the endpoints of all intervals [D[v]-X[v], D[v]]
def addRemoveSequenceTurbo(H, X, startEdge):
    d, ans = H.extendedDijkstra(X, H.verticesOf[startEdge])
    ans.reverse()

    if settings.debug:
        #distances are decreasing as we see vertices in the order for the first time.
        curMin = d[ans[0]]
        observed = set()
        for u in ans:
            if u in observed: continue
            observed.add(u)
            assert curMin >= d[u]
            curMin = d[u]

    return ans
    # endD = H.dijkstra(X, H.verticesOf[startEdge])
    #startD = {v : 0 if v in H.verticesOf[startEdge] else min(endD[u] for u in H.G[v]) - (settings.epsilon*settings.epsilon) for v in H.V}
    #pairList = [(endD[v], v) for v in H.V] + [(startD[v], v) for v in H.V]
    #pairList.sort()
    #return [v for (_, v) in reversed(pairList)]

