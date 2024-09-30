import os
import random
import pickle
import itertools

class UnionFind:
    def __init__(self, n):
        '''
        args:
            n:int = number of nodes in the union find data structures. Nodes are index 
                by integers between 0 and n-1
        '''
        self.n = n
        self.parents = [i for i in range(n)]
        self.rank = [1]*n
    
    def find(self, i):
        '''
        args:
            i:int = index of some node
        returns:
            an integer representing the root of the set in which node i belongs
        '''
        assert i >= 0 and i <= self.n-1, f"Node {i} is not in the data structure. Only nodes {0} through {self.n-1} exist."
        if i != self.parents[i]:
            self.parents[i] = self.find(self.parents[i])
        return self.parents[i]
    
    def union(self, pi, pj):
        '''
        args:
            pi:int = index of some parent node
            pj:int = index of some parent node    
        '''
        assert pi >= 0 and pi <= self.n-1, f"Node {pi} is not in the data structure. Only nodes {0} through {self.n-1} exist."
        assert pj >= 0 and pj <= self.n-1, f"Node {pj} is not in the data structure. Only nodes {0} through {self.n-1} exist."

        pi = self.find(pi)
        pj = self.find(pj)
        if pi != pj:
            self.parents[pi] = pj

# https://stackoverflow.com/questions/9557182/python-shuffle-only-some-elements-of-a-list
def shuffle_slice(a, start, stop):
    i = start
    while (i < stop-1):
        idx = random.randrange(i, stop)
        a[i], a[idx] = a[idx], a[i]
        i += 1

# type LeafCost = dict[int, tuple[list[int], float]]
def metric_tsp_approximation(matrix, block_size = 24):
    """
    An algorithm for solving the Metric TSP using minimum spanning trees and depth first search.
    
    Args:
        matrix: List[List[float]]
            An n x n matrix of distances, where M[i, j] corresponds to the distance from city i to city j.
    
    Returns:
        path: List[int] 
            A list corresponding to the order in which to visit cities, starting from path[0] and ending 
            at path[-1] before returning to path[0].  
    """
    n = len(matrix)
    mst = [[] for _ in range(n)]
    uf = UnionFind(n)
    edges = []
    for i in range(n):
        for j in range(n):
            edges.append((matrix[i][j], i, j))
    edges.sort()
    for i in range(0, len(edges), block_size):
        shuffle_slice(edges, i, min(i+block_size, len(edges)))
        shuffle_slice(edges, i+int(block_size/2), min(i+int(block_size*3/2), len(edges)))

    for e in edges:
        u,v = e[1],e[2]
        if uf.find(u) != uf.find(v):
            # print(u, v, uf.parents)
            uf.union(u,v)
            mst[u].append(v)
            mst[v].append(u)
    # print(mst)

    def join_leaf_cost(leaf_cost, next_start: int):
        join_min: float = math.inf
        ret1 = []
        for last_u, (arr, cost) in leaf_cost.items():
            # print(": ", cost, next_start, join_min, len(arr))
            if cost + matrix[last_u][next_start] < join_min:
                join_min = cost + matrix[last_u][next_start]
                ret1 = arr
        return ret1, join_min

    # return map(last_u, (arr, cost)), aka leaf_cost
    def dfs(u) -> dict[int, tuple]:
        # print(u)
        vis[u] = True
        ret = dict()
        leaf_costs = []
        for v in mst[u]:
            # print(":", v, mst[u])
            if not vis[v]:
                leaf_costs.append((v, dfs(v)))
        if len(leaf_costs) == 0:
            return {u: ([u], 0)}
        if len(leaf_costs) > 5:
            all_permu = [leaf_costs[:] for _ in range(100)]
            for val in all_permu:
                random.shuffle(val)
        else:
            all_permu = itertools.permutations(leaf_costs) 
        rep_cnt = 0
        for permu in all_permu:
            rep_cnt += 1
            if rep_cnt > 120:
                assert False
            # print(permu)
            cur_cost = matrix[u][permu[0][0]]
            cur_arr = [u]
            for i in range(len(permu)-1):
                v, c = join_leaf_cost(permu[i][1], permu[i+1][0])
                cur_arr += v
                cur_cost += c
            for last_u, (arr, cost) in permu[-1][1].items():
                if last_u not in ret or cur_cost + cost < ret[last_u][1]:
                    ret[last_u] = (cur_arr + arr, cur_cost + cost)
        return ret

    ans_x = math.inf
    ans = []

    for i in range(n):
        vis = [False]*n
        final_leaf_cost = dfs(i)
        cur_ans, cur_x = join_leaf_cost(final_leaf_cost, i)
        # new_x = validate_tour(cur_ans, matrix)
        # assert abs(cur_x - new_x) < 0.01
        if cur_x < ans_x:
            ans_x = cur_x
            ans = cur_ans
        # for adj in mst:
        #     random.shuffle(adj)
        # if new_x < ans_x:
        #     ans_x = new_x
        #     ans = ret
    # print(i, ans_x)
    return ans, ans_x

import math
def tsp_greedy(matrix, home):
    """
    A greedy implementation of TSP, starting and ending at home.
    
    Args:
        matrix: List[List[float]]
            An n x n matrix of distances, where M[i, j] corresponds to the distance from city i to city j.
        home: int
            The index of the city to start and end at.
    
    Returns:
        path: List[int] 
            A list corresponding to the order in which to visit cities, starting from path[0] and ending 
            at path[-1] before returning to path[0]. path[0] should be home.    
    """
    cur = home
    cities = set(range(len(matrix)))
    cities.remove(home)
    ret = [home]
    while len(cities) > 0:
        target = 0
        min_len = math.inf
        for c in cities:
            if matrix[cur][c] < min_len:
                target = c
                min_len = matrix[cur][target]
        cur = target
        cities.remove(target)
        ret.append(target)
    return ret

def tsp_greedy_general(matrix, cur_best):
    ret = tsp_greedy(matrix, 0)
    n = len(matrix)
    for i in range(0, len(matrix)):
        new_tour = tsp_greedy(matrix, i)
        if False:
            improve(new_tour, matrix, 10)
        if validate_tour(new_tour, matrix) < validate_tour(ret, matrix):
            ret = new_tour
    return ret

def improve_once(tsp, matrix, cur_best):
    x = validate_tour(tsp, matrix)
    assert x == cur_best
    n = len(tsp)
    def get_cost(pos1, pos2):
        ret = 0
        if pos1 != 0: ret += matrix[tsp[pos1-1]][tsp[pos1]]
        else: ret += matrix[tsp[-1]][tsp[0]]
        ret += matrix[tsp[pos1]][tsp[pos1+1]]
        if pos2 != pos1 + 1: ret += matrix[tsp[pos2-1]][tsp[pos2]]
        if pos2 != n-1: ret += matrix[tsp[pos2]][tsp[pos2+1]]
        else: ret += matrix[tsp[-1]][tsp[0]]
        return ret

    for i in range(n-1):
        for j in range(i+1,n):
            x1 = get_cost(i, j)
            tsp[i], tsp[j] = tsp[j], tsp[i]
            x2 = get_cost(i, j)
            # x = validate_tour(tsp, matrix)
            # assert x == cur_best + x2 - x1
            x = cur_best + x2 - x1
            if x < cur_best:
                # return x, tsp
                cur_best = x
            else:
                tsp[i], tsp[j] = tsp[j], tsp[i]
    return cur_best, tsp

def improve(tsp, matrix, cnt=20):
    cur_best = validate_tour(tsp, matrix)
    last_best = cur_best
    for _ in range(cnt):
        cur_best, tsp = improve_once(tsp, matrix, validate_tour(tsp, matrix))
        if last_best == cur_best: break
        # print(cur_best)
        last_best = cur_best
    return tsp, cur_best

TOTAL_TIME = 2400
TIME_PER_CASE = 12

import sys
import time
import multiprocessing
def improved_tsp_approximation(matrix):
    outfile=sys.stdout
    random.seed(0)
    # print(multiprocessing.cpu_count())

    start_time = time.perf_counter()
    n = len(matrix)
    tsp, tsp_val = [], math.inf
    # greedy = tsp_greedy_general(matrix, math.inf)
    # greedy, _ = improve(greedy, matrix)
    # print("Greedy time", time.perf_counter() - start_time)
    r = 0
    while True:
        r+=1
        block_size = random.choice([48, 60, 128, 256, 512])
        cur, cur_val = metric_tsp_approximation(matrix, block_size)
        cur, cur_val = improve(cur, matrix)
        # print(block_size, cur_val, file=outfile)
        if cur_val < tsp_val:
            tsp = cur
            tsp_val = cur_val
        if time.perf_counter() - start_time > TIME_PER_CASE:
            break
    print("Time  :\t", time.perf_counter() - start_time)
    print("repeat:\t", r)
    tsp, _ = improve(tsp, matrix)

    print("mst   :\t", validate_tour(tsp, matrix), file=outfile)
    # print("greedy:\t", validate_tour(greedy, matrix), file=outfile)
    print()
    ret = tsp
    ans = math.inf
    for algo in [tsp]:
        new_ans = validate_tour(algo, matrix)
        if new_ans < ans:
            ans = new_ans
            ret = algo
    return ret


def validate_tour(tour, matrix):
    """
    Provided function to verify the validity of your TSP approximation function.
    Returns the length of the tour if it is valid, -1 otherwise.
    Feel free to use or modify this function however you please,
    as the autograder will only call your tsp_approximation function.
    """
    n = len(tour)
    cost = 0
    for i in range(n):
        if matrix[tour[i - 1]][tour[i]] == float("inf"):
            return -1
        cost += matrix[tour[i - 1]][tour[i]]
    return cost


def verify_basic(matrix, path):
    """Verify that the proposed solution is valid."""
    assert len(path) == len(
        matrix
    ), f"There are {len(matrix)} cities but your path has {len(path)} cities!"
    assert sorted(path) == list(
        range(len(path))
    ), f"Your path is not a permutation of cities (ints from 0 to {len(path)-1})"


import concurrent.futures
def evaluate_tsp(tsp_approximation):
    """
    Provided function to evaluate your TSP approximation function.
    Feel free to use or modify this function however you please,
    as the autograder will only call your tsp_approximation function.
    """

    test_cases = pickle.load(open(os.path.join("tsp_cases.pkl"), "rb"))

    total_cost = 0

    # all_cases = test_cases["files"] + test_cases["generated"]
    # # all_cases = all_cases[:6]
    # with concurrent.futures.ProcessPoolExecutor(max_workers=12) as executor:
    #     results = executor.map(tsp_approximation, all_cases)
    #
    # for case, tour in zip(all_cases, results):
    #     # tour = tour.
    #     # print("CASE: ", len(case), type(case))
    #     # print("TOUR: ", len(tour), tour)
    #     verify_basic(case, tour)
    #     cost = validate_tour(tour, case)
    #     total_cost += cost
    # print(f"Total cost: {total_cost}")
    # return total_cost

    for i, case in enumerate(test_cases["files"] + test_cases["generated"]):
        if i > 5: continue
        # if len(case) > 150: continue
        # if i != 50: continue
        # case = [
        #         [0, 4, 5],
        #         [4, 0, 6],
        #         [5, 6, 0],
        #         ]
        print(f"starting case {i} with len:", len(case))
        tour = tsp_approximation(case)
        verify_basic(case, tour)
        cost = validate_tour(tour, case)
        assert cost != -1
        total_cost += cost
        print(f"Case {i}: {cost}")

    print(f"Total cost: {total_cost}")
    return total_cost


if __name__ == "__main__":
    evaluate_tsp(improved_tsp_approximation)
