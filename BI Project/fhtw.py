import argparse
import time
import hypertreeDecomposition
import sys
from hypergraph import *
import os
import statistics
import traceback
import settings

def runDecompositions(H, name, numberOfRuns, timeLimit):
    decompositionResults = []
    best_decomp_width = float('inf')
    best_decomp_index = -1
    best_lower_bound = 0
    best_lower_bound_index = -1

    start_time = time.time()
    elapsed_time = 0
    average_time = 0

    # order is 4 -> 0 -> 2 -> 3
    # cutoffs are Heuristic: 10, Normal: 20, Boosted: 20, ILP: 10
    if settings.doMixedRun:
        numberOfRuns = settings.num_heuristic + settings.num_normal + settings.num_boosted + settings.num_ILP
    
    for i in range(numberOfRuns):
        if settings.doMixedRun:
            if i < settings.num_heuristic: 
                # Heuristic
                settings.useSeparatorLP = settings.availableSeparatorLPs[4]
            elif settings.num_heuristic <= i < settings.num_heuristic + settings.num_normal: 
                # Normal
                settings.useSeparatorLP = settings.availableSeparatorLPs[0]
            elif settings.num_heuristic + settings.num_normal <= i < settings.num_heuristic + settings.num_normal + settings.num_boosted:
                # Boosted
                settings.useSeparatorLP = settings.availableSeparatorLPs[2]
            else:
                # ILP
                settings.useSeparatorLP = settings.availableSeparatorLPs[3]

        try:
            outputDec, stats = hypertreeDecomposition.runDecomposition(H)
            outputDec.verifyDecomposition(H)
        except Exception as e:
            print(f'[{time.time()}]: Caught exception {e} for {name}, run {i}', file=sys.stderr)
            traceback.print_exc(file=sys.stderr)
            continue

        decompositionResults.append((outputDec, stats))

        time_this_run_took = (time.time() - start_time) - elapsed_time
        elapsed_time += time_this_run_took
        average_time = elapsed_time / (i + 1)

        for _ in range(10):
            if not os.path.exists(f'Output/{name}'):
                try:
                    os.makedirs(f'Output/{name}')
                except Exception as e:
                    print(f'[{time.time()}]: Caught exception {e} for {name}, run {i}', file=sys.stderr)
                    print(f'OS Path Exists: {os.path.exists(f"Output/{name}")}', file=sys.stderr)
                    traceback.print_exc(file=sys.stderr)
                else:
                    break

        for _ in range(10):
            try:
                writeToFile = open(f'Output/{name}/run_{i}.out', 'w')
            except Exception as e:
                print(f'[{time.time()}]: Caught exception {e} for {name}, run {i}', file=sys.stderr)
                print(f'OS Path Exists: {os.path.exists(f"Output/{name}")}', file=sys.stderr)
                traceback.print_exc(file=sys.stderr)
            else:
                break
        
        print(outputDec, file = writeToFile)
        writeToFile.write('\n\n')
        statistics.printStatistics(writeToFile)
        writeToFile.write(f'Total time spent solving: {time_this_run_took:.3f}')
        writeToFile.write('\n\n')

        writeToFile.write(f'LP Solver: {settings.useSeparatorLP}')
        writeToFile.write('\n')
        writeToFile.write(f'Number of Rounding Attemps: {settings.roundingAttempts}')
        writeToFile.write('\n')
        writeToFile.write(f'Lower Bound Witness = {outputDec.fhtwLowerWitness}')

        writeToFile.close()

        if (outputDec.fhtwUpper < best_decomp_width):
            best_decomp_width = outputDec.fhtwUpper
            best_decomp_index = i

        if (outputDec.fhtwLower > best_lower_bound):
            best_lower_bound = outputDec.fhtwLower
            best_lower_bound_index = i

        if elapsed_time > timeLimit:
            break
        
    return decompositionResults, elapsed_time, best_decomp_width, best_decomp_index, best_lower_bound, best_lower_bound_index

def main():
    parser = argparse.ArgumentParser(
                    prog='fhtw',
                    description='Computes an approximate fhtw of a graph')
    
    parser.add_argument('name', type=str, help='The hyperbench name of the hypergraph')
    parser.add_argument('-n', metavar='numRuns', type=int, default=10, help='The number of times to run the algorithm')
    parser.add_argument('-t', metavar='timeLimit', type=int, default=600, help='The time limit of the algorithm, in seconds')
    
    args = parser.parse_args()

    name, numberOfRuns, timeLimit = args.name, args.n, args.t

    sys.setrecursionlimit(100000) 

    # if not ('wiki' in name or 'tpcds' in name or ('tpch' in name and'tpch-' not in name)):
    #     sys.stdin = open(f'Hypergraphs/{name}')
    # else:
    #     sys.stdin = open(f'New_Instances/{name}')

    sys.stdin = open(f'Hypergraphs/{name}')

    H = hypergraph()
    print(f'[{time.time()}]: Starting fhtw on {name}')

    start_time = time.time()

    decompositionsResults, elapsed_time, best_width, best_index, best_lb, best_lb_index = runDecompositions(H, name, numberOfRuns, timeLimit)

    end_time = time.time()
    output_folder = f'Output/{name}'
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    with open(f'Output/{name}/summary.out', 'w') as f:
        f.write(f'Ran {len(decompositionsResults)} out of {numberOfRuns} decompositions, using {elapsed_time:.2f} out of {timeLimit} seconds\n\n')
        f.write(f'All decomposition widths = {[d.fhtwUpper for (d, s) in decompositionsResults]}\n\n')
        f.write(f'best width = {best_width}, which is decomposition index {best_index}\n')
        f.write(f'best lower bound so far is {best_lb}, which is decomposition index {best_lb_index}\n\n')
        f.write(f'Total time taken: {end_time - start_time:.2f} seconds')

    print(f'[{time.time()}]: Finished running {name} with best decomposition width {best_width} (index {best_index}); ran {len(decompositionsResults)} out of {numberOfRuns} decompositions, using {elapsed_time:.2f} out of {timeLimit} seconds')

if __name__ == '__main__':
    main()