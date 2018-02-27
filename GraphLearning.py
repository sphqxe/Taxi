import csv
from math import pi,atan2
from shapely.geometry import Point, LineString
from tqdm import tqdm as tn
import json

# Computes the angle between 2 line segments
def ag(p1, p2):
    return atan2((p2[1]-p2[1]),(p2[0]-p2[0]))

# Generator function to traverse a graph in a breadth-first manner
# Parameters are: Graph (Dictionary), Starting Node, Depth Limit, Starting Route (Initialize with empty tuple)
# Returns Tuples representing routes to reach that graph node
def BFTraverse(G,N,D,S):
    yield S+(N,)
    if D:
        for adjNode in G[N]:
            yield from BFTraverse(G,adjNode,D-1,S+(N,))

# Euclidean distance
def eu(x1,y1,x2,y2):
    return (((x1-x2)**2)+((y1-y2)**2))**0.5

# Training Data File Input
with open("train_data.csv") as f:
    d = list(csv.reader(f, delimiter=",", quotechar='"'))
    d = [{'id': int(i[0]),\
          'taxiId': int(i[1]),\
          'timeStamp': i[2],\
          'duration': int(i[3]),\
          'startPos':(int(i[4]), int(i[5])),\
          'endPos':(int(i[6]), int(i[7])),\
          'traj':list(zip([int(j) for j in i[8].split(',')], [int(j) for j in i[9].split(',')]))\
         } for i in d[1:]]

# Sort by number of data points in trajectory so the longer trajectories get laid on the map first
d.sort(key=lambda x: len(x['traj']), reverse=True)

# Distance limit parameter - This parameter specifies the maximum perpendicular distance between an edge E and a node N for merging to be considered
dLim = 5
# Direction limit parameter - This parameter specifies the maximum difference in direction of an edge E and a node N for merging to be considered
# The direction of node N is defined by the vector D->E if D->N->E are sequential nodes in the graph
rLim = 5*pi/180

G = {}
V = {}

# for tripIdx in tn(range(len(d))):
for tripIdx in tn(range(5000)):
    trip = d[tripIdx]
    correctedTraj = []
    prevNode = None
    for nodeIdx in range(len(trip['traj'])):
        n = trip['traj'][nodeIdx]
        if nodeIdx>0 and n == trip['traj'][nodeIdx-1]:
            continue
        mergeFlag = False
        for i in G:
            if abs(i[0]-n[0])>20 or abs(i[1]-n[1])>20:
                continue
            for j in G[i]:
                if abs(j[0]-n[0])>20 or abs(j[1]-n[1])>20:
                    continue
                N = Point(*n)
                E = LineString([i,j])
                if N.distance(E) < dLim and\
                nodeIdx > 0 and nodeIdx < len(trip['traj'])-1 and\
                ag(trip['traj'][nodeIdx-1], trip['traj'][nodeIdx+1]) < rLim:
                    mergeFlag = True
                    v = i if Point(*i).distance(N) < Point(*j).distance(N) else j
                    if correctedTraj and correctedTraj[-1]!=v or not correctedTraj:
                        correctedTraj.append(v)
                    if prevNode:
                        newEdgeFlag = True
                        for route in BFTraverse(G,prevNode,3,tuple()):
                            if route[-1]==v:
                                newEdgeFlag = False
                                pNode = prevNode
                                for k in range(len(route)-1): # The trip ID has to be added to every single edge on the route from the previous node to the merged node
                                    V.setdefault((pNode,route[k]),set()).add(trip['id'])
                                    pNode = route[k]
                                break
                        if newEdgeFlag:
                            G.setdefault(prevNode,set()).add(v)
                            V.setdefault((prevNode,v),set()).add(trip['id'])
                    prevNode = v
        if not mergeFlag:
            if correctedTraj and correctedTraj[-1]!=n or not correctedTraj:
                correctedTraj.append(n)
            G.setdefault(n,set())
            if prevNode:
                G[prevNode].add(n)
                V.setdefault((prevNode,n),set()).add(trip['id'])
            prevNode = n
    d[tripIdx]['corrTraj'] = correctedTraj

with open('learnedGraph.json') as f:
    f.write(json.dumps(G))

with open('edgeVolume.json') as f:
    f.write(json.dumps(V))
