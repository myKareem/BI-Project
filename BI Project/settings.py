import pulp


### General settings

#debug mode?
debug = False

#tolerance threshold
epsilon = 0.000001

# whether to print tree decomposition as it is found to sys.stderr.
printProgress = False

#when fhtw upper bound grows >= kill switch, stop looking.
killSwitch = 1000

### Solver settings

def getSolver():
    #Which solver to use?
    #Note: I had do jerry-rig the PuLP API for HiGHS for the msg=false thing to work. There is also a "callback" thing that 
    #does not work and i had to comment out the offending line in the API. 
    # solver = pulp.HiGHS(msg=False)
    # Stop asking for Gurobi. Use the default solver (CBC) included with PuLP.
    return pulp.PULP_CBC_CMD(msg=False)
    

    #Gurobi - not open source.
    #solver = pulp.GUROBI(msg=False, timeLimit=3600)

    # solver = pulp.HiGHS(msg=False, timeLimit=3600)


    #################################

    #barely slower than the library one
    #prob.solve(pulp.HiGHS_CMD(msg=False, path=r'C:\Users\danie\Dropbox\pprog\hypertreewidth\highs.exe'))
        
    #works, slower
    #prob.solve(pulp.GLPK_CMD(msg=False, path=r'C:\Users\danie\Dropbox\pprog\hypertreewidth\CLP\bin\glpsol.exe'))   
 
    #works, slower
    #prob.solve(pulp.SCIP_CMD(msg=False, path=r'C:\Program Files\SCIPOptSuite 9.0.0\bin\scip.exe'))

    #works, slower
    #prob.solve(pulp.PULP_CBC_CMD(msg=False))
    
    #return solver


# ------------------------------------------------------------------------------------


# Warm start settigns (to add?)
#warm start with random set of vertices, or random set weighted by fractional cover, or specific bag?


# Fractional Balanced Separator Settings ------------

# GaifmanNormal and HypergraphNormal seem to give the same optimal LP values
# But GaifmanBoosted one is slightly slower while giving smaller decompositions?? Maybe confirmation bias... 
#
# Using GaifmanNormal is a lot faster!
#
# Decide which separator LP to use. 
availableSeparatorLPs = ["GaifmanNormal", "HypergraphNormal", "GaifmanBoosted", "GaifmanNormalILP", "gammaPlusFractionalCoverHeuristic"]

#useSeparatorLP = availableSeparatorLPs[4]
useSeparatorLP = "HypergraphNormal"
doMixedRun = False        # doesn't matter what useSeparatorLP is here, nor -n numRuns passed in fhtw command
num_heuristic = 0
num_normal = 20
num_boosted = 0
num_ILP = 0

### Rounding settings ------------

# How many times to round same LP. Only makes sense to be > 1 when candidateSetRoot = randomNonzeroGammaEdge
# Does not matter for ILP.
# Set to at least 10 for good results.
roundingAttempts = 10

# Set to true to use estimator for fractional cover of Z inside current component when rounding. Faster but less accurate.
useZCoverEstimator = True

# Set to true to use loss estimator instead of true gain when rounding. Faster but less accurate.
useLossEstimator = True

#set to True to only use candidate sets that differ at least epsilon significantly in time to the previous one. 
#Faster, but less accurate.
pruneCandidateSets = True

# set to true to use Turbo Rounding mode. This implies that useZCoverEstimator = True and useLossEstimator = True and pruneCandidateSets = False
turboRounding = True


#when rounding, where to start Disktra from. allNonzeroGammaEdge is slower but might give better results (?)
#looks like randomNonzeroGammaEdge + roundingAttempts >= 5 gives best results so far. 
possibleCandidateRootSets = ["heaviestGammaEdge", "randomNonzeroGammaEdge", "allNonzeroGammaEdge"]
candidateSetRoot = possibleCandidateRootSets[1]

#how much to give priority to separators that decrease a lot the size of the largest component. 
#1 means care about component size only, 0 means care about splitting root only. 0 is default.
#++could be worth adding a similar metric for reducing the fractional cover of the large component instead of the size?

#need to do statistical analysis of how well it does
vertexGainWeight = 0

# in gammaPlusFractionalCoverHeuristic, how much weight to give to the fractional cover of all vertices vs current root bag. 
# Note that this stacks with vertexGainWeight when rounding, so perhaps set vertexGainWeight = 0 when using the gammaPlusFractionalCoverHeuristic
heuristicFCovWeight = 10


### Lower bound settings --------------------------------------------------------------------------------------
lowerBoundNumVertices = 30
lowerBoundNumTries = 10
