#
#The current version of pulp seems to have an error with callbacks. We dont need those anyway so just comment out the offending line in the HiGHS api.
#
#Also needed to add a line in the HiGHS api to set the console output flag to false to avoid lots of console output. 
#

from queue import PriorityQueue
from collections import deque
import re

class hypergraph:
    
    #
    # default constructor, fills fields from stdin:
    # (U, F) -- universe of discourse and set of possible hyperedges. "Pass by reference"
    # vName and eName are names of vertices/hyperedges, passed by reference too.
    # V and E are actual vertex and hyperedge edge sets (sets of ID numbers pointing to (U, F)). Copied on construction. 
    # edgesOf and verticesOf are vertex edge incidences for V and E, copied and trimmed on construction. 
    # 
    #
    def __init__(self, *args):
        if(len(args) == 0):
            self.U, self.vName, self.edgesOf, self.F, self.eName, self.verticesOf = self.readInput()
            self.V = set(self.U)
            self.E = set(self.F)
            self.G = self.gaifmanGraph()
        else:
            self.U = args[0]
            self.F = args[1]
            self.V = set(args[2])
            self.E = set(args[3])
            self.vName = args[4]    
            self.eName = args[5]
            self.edgesOf = {v : args[6][v] & self.E for v in self.V} 
            self.verticesOf = {e : args[7][e] & self.V  for e in self.E}
            self.G = {v : args[8][v] & self.V for v in self.V}
        
        for e in self.E: 
            for v in self.verticesOf[e]: assert e in self.edgesOf[v], "edge not in edge list of one of its vertices"
        
        for v in self.V:
            for e in self.edgesOf[v]: assert v in self.verticesOf[e], "v not in vertex list of one of its edges"
    
    # Reads hypergraphs on format used by http://hyperbench.dbai.tuwien.ac.at/ from stdin.
    def readInput(self):    
        vertexIDs = []
        edgesOf = {}
        vName = {}
        vID = {}
    
        edgeIDs = []
        verticesOf = {}
        eName = {}
        eID = {}
    
        while True:
            inputLine = input().strip()
            if inputLine == "" or inputLine[0] == '%': continue

            lines = []
            if inputLine.count('(') > 1:
                lines = inputLine.split('),')
            else:
                lines = [inputLine]
            
            for line in lines:
                tokens = re.split(r'[\(\)\,]', line)
                tokens = [s.strip() for s in tokens if s.strip() != ""]
            
                edge = tokens[0]
                
                eID[edge] = len(edgeIDs)
                edgeIDs.append(eID[edge])
                eName[eID[edge]] = edge
                verticesOf[eID[edge]] = set()
                
                for vertexToken in tokens[1:]:
                    if vertexToken == ".":
                        return set(vertexIDs), vName, edgesOf, set(edgeIDs), eName, verticesOf
                
                    if vertexToken not in vID:
                        vID[vertexToken] = len(vertexIDs)
                        vertexIDs.append(vID[vertexToken])
                        vName[vID[vertexToken]] = vertexToken
                        edgesOf[vID[vertexToken]] = set()
                        
                    verticesOf[eID[edge]].add(vID[vertexToken])
                    edgesOf[vID[vertexToken]].add(eID[edge])
                
        assert False, "EOF reached without . token at end of line!"
        
    def gaifmanGraph(self):
        ans = {v : set() for v in self.V}
        for e in self.E:
            for u in self.verticesOf[e]:
                for v in self.verticesOf[e]:
                    if v == u: continue
                    ans[u].add(v)
                    ans[v].add(u)
        return ans            

    # returns all hyperedges in eset that are incident to at least one vertex of vset.
    def incidentHyperedges(self, *args):
        assert len(args) in [1,2], "incidentHyperedges expecting one or two positional arguments"
        
        vset = args[0]
        eset = args[1] if len(args) == 2 else self.E
    
        vsetsize = sum(len(self.edgesOf[v]) for v in vset)
        esetsize = sum(len(self.verticesOf[e]) for e in eset)
        ans = set()

        if vsetsize < esetsize:
            for v in vset: ans.update(self.edgesOf[v] & eset)
        else:
            for e in eset:
                for v in self.verticesOf[e]:
                    if v in vset:
                        ans.add(e)
                        break
        return ans
        
    # returns all vertices in vset that are incident to at least one hyperedge in eset.
    def incidentVertices(self, *args):
        assert len(args) in [1,2], "incidentVertices expecting one or two positional arguments"
        
        if len(args) == 1:
            vset = self.V
            eset = args[0]
        else:
            vset = args[0]
            eset = args[1]
    
        vsetsize = sum(len(self.edgesOf[v]) for v in vset)
        esetsize = sum(len(self.verticesOf[e]) for e in eset)
        ans = set()

        if esetsize < vsetsize:
            for e in eset: ans.update(self.verticesOf[e] & self.V)
        else:
            for v in self.V:
                for e in self.edgesOf[v]:
                    if e in eset:
                        ans.add(v)
                        break
        return ans
  
    def subhypergraph(self, vset, eset):
        return hypergraph(self.U, self.F, vset, eset, self.vName, self.eName, self.edgesOf, self.verticesOf, self.G)

    def inducedSubhypergraph(self, vset):
        return self.subhypergraph(vset, self.incidentHyperedges(vset))

    #returns connected components of (vset, eset) as a list of sets of vertices.    
    def connectedComponents(self):    
        visited = set()
        ans = []
        for v in self.V:
            if v in visited: continue
            ans.append(self.BFS(v))
            visited.update(ans[-1])
        
        return ans
        
    #bfs in the sub-hypergraph. 
    def BFS(self, pos):
        ans = set()
        Q = deque()
        Q.append(pos)
        while Q:
            v = Q.popleft()
            if v in ans: continue
            ans.add(v)
            for u in self.G[v]:
                if u not in ans:
                    Q.append(u)
        return ans
    
    #sources is a set of vertices, to start Dijkstra from. 
    #X is dict : vertex, vertex --> edge cost.  Edges are directional. X(v,v) is the cost to start at v.
    def dijkstra(self, X, sources):
        D = {v : (X[(v, v)] if v in sources else float('inf')) for v in self.V} 
        visited = set()
        Q = PriorityQueue()
        for v in sources: Q.put((D[v], v))
        while(not Q.empty()):
            d, v = Q.get()
            if v not in visited: 
                visited.add(v)
                for u in self.G[v]:
                    if D[v] + X[(v,u)] < D[u]:
                        D[u] = D[v] + X[(v,u)]
                        Q.put((D[u], u))
        return D
    

    #sources is a set of vertices, to start Dijkstra from. 
    #X is dict : vertex, vertex --> edge cost.  Edges are directional. X(v,v) is the cost to start at v.
    #returns distances from sources + an ordering of vertices where every vertex appears as it is visited,
    #and then again when all neighbors of the vertices become visited. 
    def extendedDijkstra(self, X, sources):
        D = {v : (X[(v, v)] if v in sources else float('inf')) for v in self.V} 
        visited = set()

        addRemoveSequence = sorted(list(sources), key = lambda v : X[(v,v)])
        discovered = set(sources)

        Q = PriorityQueue()
        for v in sources: Q.put((D[v], v))
        while(not Q.empty()):
            d, v = Q.get()
            if v not in visited: 
                visited.add(v)
                for u in self.G[v]:
                    if D[v] + X[(v,u)] < D[u]:
                        if u not in discovered:
                            discovered.add(u)
                            addRemoveSequence.append(u)
                        D[u] = D[v] + X[(v,u)]
                        Q.put((D[u], u))
                addRemoveSequence.append(v)
        assert len(addRemoveSequence) == 2*len(self.V)
        
        return D, addRemoveSequence

        
    # returns the closed neighborhood of S.
    def closedNeighborhood(self, S):
        ans = set(S)
        for v in S: ans.update(self.G[v])
        return ans
        
    # returns the open neighborhood of S. 
    def openNeighborhood(self, S):
        return self.closedNeighborhood(S) - S
    
    #new as of 12/6/2024. 
    #returns the sub-hypergraph of H induced by the inclusion-maximal hyperedges
    def trimmedHypergraph(self):
        neset = set()
        for e in self.E:
            #print("trimming ", self.eName[e])
            incV = self.verticesOf[e]
            neighborEdges = self.incidentHyperedges(incV)
            #print([self.eName[q] for q in neighborEdges])
            if not any(self.verticesOf[e] < self.verticesOf[ee] for ee in neighborEdges): 
            #    print("include ", self.eName[e])
                neset.add(e)
        return self.subhypergraph(self.V, neset)
        
    # for output. 
    def __str__(self):
        ans = []
        
        for e in self.E:
            ans.append(self.eName[e] + "(" + str(e) + ") : " + [self.vName[v] for v in self.verticesOf[e]].__str__())
            
        ans.append('\nGaifmanGraph\n')
            
        for v in self.G:
            ans.append(self.vName[v] + " (" + str(v) + "): " + [self.vName[u] for u in self.G[v]].__str__()) 
            
        return '\n'.join(ans)
                      
    def printVertexSet(self, vset):
        return [self.vName[v] for v in vset].__str__()

    def printEdgeSet(self, eset):
        return [self.eName[e] for e in eset].__str__()
        
    

    
  
