import sys

#max number of edges with non-zero Y
maxNonZeroY = 0

#max number of times we go around the outer loop when rounding
maxRoundingLoops = 0

#max number of candidate sets
maxCandidateSets = 0

#max number of components that an edge e with non-zero y_e intersects the neighborhood of during rounding. 
maxComponentNeighborhoodTouchings = 0


#total time rounding
totalTimeRoundingBalSepLP = 0

#total time solving balanced separator LP
totalTimeSolvingBalSepLP = 0

#total time solving balanced separator LP
totalTimePreprocessing = 0



def printStatistics(file=sys.stderr):
    print("Max number of edges with non zero Y: %d" % maxNonZeroY, file=file)
    print("Max number of iterations of outer rounding loop: %d" % maxRoundingLoops, file=file)
    print("Max number of candidate sets: %d" % maxCandidateSets, file=file)
    print("Time spent rounding: %.3f" % totalTimeRoundingBalSepLP, file=file)
    print("Time spent solving Separator LP: %.3f" % totalTimeSolvingBalSepLP, file=file)
    # print("Time spent preprocessing: %.3f" % totalTimePreprocessing, file=file)

def getStatistics():
    return {
        'maxNonZeroY': maxNonZeroY,
        'maxRoundingLoops': maxRoundingLoops,
        'maxCandidateSets': maxCandidateSets,
        'totalTimeRoundingBalSepLP': totalTimeRoundingBalSepLP,
        'totalTimeSolvingBalSepLP': totalTimeSolvingBalSepLP,
        'totalTimePreprocessing': totalTimePreprocessing
    }

def resetStatistics():
    global maxNonZeroY, maxRoundingLoops, maxCandidateSets, maxComponentNeighborhoodTouchings, totalTimeRoundingBalSepLP, totalTimeSolvingBalSepLP, totalTimePreprocessing

    maxNonZeroY = 0
    maxRoundingLoops = 0
    maxCandidateSets = 0
    maxComponentNeighborhoodTouchings = 0
    totalTimeRoundingBalSepLP = 0
    totalTimeSolvingBalSepLP = 0
    totalTimePreprocessing = 0
    

