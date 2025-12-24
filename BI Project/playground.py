from hypergraph import *
from fractionalBalancedSeparator import *
from balancedSeparator import *
import settings
import time
import random

H = hypergraph()

#H.printMe()
#H.printVertexSet(H.V)

largestVal = -1
worstVset = None

for tries in range(settings.lowerBoundNumTries):
    start_time = time.time()

    vset = {random.choice(list(H.V)) for _ in range(settings.lowerBoundNumVertices)}
    eset = {next(iter(H.edgesOf[v])) for v in vset}
    gamma = {e : 1 if e in eset else 0 for e in H.E}
    
    X, Y, optLP = fractionalBalancedSeparator(H, gamma)

    time_taken = time.time() - start_time
      
    if optLP > largestVal:
        largestVal = optLP
        worstVset = vset 
    
    print("--- Attempt %d --- %.3f seconds ---" % (tries, time_taken))
    print("%.3f <= fhtw for vertex set:" % (optLP))
    
    for i in range(10):
        start_time = time.time()
        
        balSep = roundSeparatorLP(H, gamma, X, Y) 

        time_taken = time.time() - start_time
        print("--- Rounding Attempt %d --- %.3f seconds ---" % (i, time_taken))

        print([H.vName[v] for v in vset])
        print("Balanced separator found:")
        print([H.vName[v] for v in balSep])
        print("Fractional Cover of Balanced Separator: %.3f" % fractionalCover(H, balSep)[0])
        
        print("Remaining vertices of set to be separated:")
        print([H.vName[v] for v in vset-balSep])
        
        print("Connected Component Sizes:")
        print([len(C) for C in H.inducedSubhypergraph(H.V - balSep).connectedComponents()])
        
        print("Intersection of vset with the components:")
        print([len(C & vset) for C in H.inducedSubhypergraph(H.V - balSep).connectedComponents()])
    
    
    
    
  
 
print("--- Highest Lower Bound Found ---")
print("%.3f <= fhtw for vertex set:" % (largestVal))
print([H.vName[v] for v in worstVset])

 
