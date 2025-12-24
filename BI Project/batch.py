import multiprocessing as mp
import time
import subprocess
import sys
import traceback

from all_graphs_batch import all_graphs_batch

import os
import argparse

def run_fhtw(name):
    print(f'[{time.time()}]: Running {name} (index {names.index(name)}) on node {nodeID}.', flush=True)
    subprocess.run(['python', 'fhtw.py', '-n 60', '-t 3600', name], stdout=f, stderr=sys.stderr)
    print(f'[{time.time()}]: Finished {name} (index {names.index(name)}) on node {nodeID}.', flush=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
                    prog='batch',
                    description='Do a batch run of fhtw')
    
    parser.add_argument('numNodes', type=int, help='The number of nodes on the compute server')
    parser.add_argument('numCoresPerNode', type=int, help='The number of cores per node on the compute server')
    parser.add_argument('nodeID', type=int, help='The id of this node')

    args = parser.parse_args()

    # make directory
    if not os.path.exists(f'Output/'):
        try:
            os.makedirs(f'Output/', exist_ok=True)
        except Exception as e:
            print(f'[{time.time()}]: Caught exception {e} for batch run', file=sys.stderr)
            print(f'OS Path Exists: { os.path.exists("Output/") }')
            traceback.print_exc(file=sys.stderr)

    f = open(f'Output/complete_summary.txt', 'a')

    # set up batch
    numCoresPerNode = args.numCoresPerNode
    numNodes = args.numNodes
    nodeID = args.nodeID

    batch = all_graphs_batch

    start_index = 0
    end_index = len(batch)

    # if nodeID == 0:
    #     start_index = 3634
    # elif nodeID == 1:
    #     start_index = 3585
    # elif nodeID == 2:
    #     start_index = 
    #     end_index = 
    # elif nodeID == 3:
    #     start_index = 
    #     end_index = 

    graph_stats = [batch[graph] for graph in batch]
    graph_stats.sort(key = lambda graph : (graph[1], graph[2], graph[0]))
    names = [graph[0] for graph in graph_stats]
            
    namesThisNodeRuns = [names[i] for i in range(start_index + nodeID, min(end_index, len(names)), numNodes)]

    # sleep
    time.sleep(2)

    print(f'Starting node {nodeID}\n', file=f, flush=True)

    start_time = time.time()

    with mp.Pool(numCoresPerNode) as p:
        p.map(run_fhtw, namesThisNodeRuns, 1)

    end_time = time.time()

    print(f'Finished node {nodeID}, total time taken: {end_time - start_time}\n', file=f, flush=True)