class Trie:

    def __init__(self):
        self.root = {'#':0}

    def insert(self, word: str) -> None:
        cur = self.root
        for idx, val in enumerate(word):
            if val not in cur:
                cur[val] = {'#' : 0}
            cur = cur[val]
            if idx == len(word) -1:
                cur['#'] = 1

    def search(self, word: str) -> bool:
        cur = self.root
        for val in word:
            if val not in cur :
                return False
            cur = cur[val]
        if cur['#']:
            return True
        else:
            return False
        
    def startsWith(self, prefix: str) -> bool:
        cur = self.root

        for val in prefix:
            if val not in cur:
                return False
            cur = cur[val]

        return True
class UnionFind:
    def __init__(self, n):
        self.parent = [i for i in range(n)]
        self.rank = [0] * n

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, x, y):
        x = self.find(x)
        y = self.find(y)

        if x != y:
            if self.rank[x] < self.rank[y]:
                self.parent[x] = y
            elif self.rank[x] > self.rank[y]:
                self.parent[y] = x
            else:
                self.parent[x] = y
                self.rank[y] += 1
def prime_sieve(n):
    is_prime = [True for _ in range(n + 1)]
    is_prime[0] = is_prime[1] = False
    for p in range(int(n**0.5)+1):
        if is_prime[p]:
            for j in range(p*p, n+1, p):
                is_prime[j] = False
    return is_prime
def isprime(a):
    if a < 2:
        return False
    for i in range(2, int(a ** 0.5 + 1)):
        if a % i == 0:
            return False
    else:
        return True
def iseven(num): return True if num%2 == 0 and num>=0 else False
def inbound(row , col, arr): return 0 <= row < len(arr) and 0 <= col < len(arr[0])
def ii(): return int(input())
def li(): return list(map(int, input().split()))
def si(): return input().strip()
def sl(): return input().split()
def mi(): return map(int, input().split())  

from collections import deque, Counter, defaultdict
from bisect import bisect_left,bisect_right
from heapq import *
# from copy import deepcopy
# from random import randint
# from math import *
# from functools import lru_cache
# @lru_cache(maxsize=None)   

d4 = [(0,1),(0,-1),(1,0),(-1,0)]
d8 =  [(0 , 1) , (0 , -1) , (-1 , 0) , (1 , 0) , (-1 , -1) , (-1 , 1) , (1 , -1) , (1 , 1)]
import sys, threading

input = lambda: sys.stdin.readline().strip()

def main():
    pass
    
if __name__ == '__main__':
    
    sys.setrecursionlimit(1 << 30)
    threading.stack_size(1 << 27)

    main_thread = threading.Thread(target=main)
    main_thread.start()
    main_thread.join()

import sys
# sys.setrecursionlimit(4000)
input = sys.stdin.readline
aphlower = 'abcdefghijklmnopqrstuvwxyz'
aphupper = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'


def solve():
    
    pass

t = 1
t = ii()
for _ in range(t):
    solve()