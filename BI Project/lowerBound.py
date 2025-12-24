from hypergraph import *
from biProjectCode import *
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
    
    balSep, optLP = computeBalancedSeparator(H, vset, 1)
    
    time_taken = time.time() - start_time

    print("--- Attempt %d --- %.3f seconds ---" % (tries, time.time() - start_time))
    print("%.3f <= fhtw for vertex set:" % (optLP))
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
    
    
    if optLP > largestVal:
        largestVal = optLP
        worstVset = vset 
 
print("--- Highest Lower Bound Found ---")
print("%.3f <= fhtw for vertex set:" % (largestVal))
print([H.vName[v] for v in worstVset])

 
