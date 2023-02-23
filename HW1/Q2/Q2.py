import math


class Methods:
    def __init__(self, n, m, e):
        self.directions = 4
        self.n = n
        self.m = m
        self.graph = e

    def dfs_util(self, v, visited):
        visited.add(v)
        for neighbour in self.graph[v]:
            if neighbour not in visited:
                self.dfs_util(neighbour, visited)

    def get_maze(self):
        to_return = []
        for i in range(self.directions):
            to_return.append([])
        return to_return

    @staticmethod
    def input():
        return tuple([int(y) for y in input().split()])

    @staticmethod
    def get_inf():
        return math.inf

    @staticmethod
    def get_e(d):
        to_return = []
        for i in range(len(d)):
            to_return.append([[], [], [], []])
        return to_return

    @staticmethod
    def add_e(e, first, second, third):
        e[first][second].append(third)
        return e

    @staticmethod
    def check_d(d, first, second, third, forth):
        return d[first][second] == d[third][forth]

    @staticmethod
    def get_ttt(n, m):
        tt = []
        for i in range(n):
            this_one = []
            for j in range(m):
                this_one.append(-1)
            if i == 0:
                tt.append(this_one)
            else:
                tt.append(this_one)
        return tt

    @staticmethod
    def get_loc(d):
        to_return = []
        for i in range(len(d)):
            to_return.append((-1, -1))
        return to_return

    @staticmethod
    def calculator(k):
        return (k + 2) % 4

    @staticmethod
    def get_fit(d):
        to_return = []
        for i in range(len(d)):
            to_return.append(None)
        return to_return

    @staticmethod
    def find(x):
        for i in x:
            if i and i is not None:
                return True
        return False


now_price = level = 0
main_one = Methods.get_inf()
n, m = [int(y) for y in input().split()]
d = []
for i in range(n * m):
    if i == 0:
        d.append(Methods.input())
    else:
        d.append(Methods.input())
e = Methods.get_e(d)
for i in range(len(d)):
    for j in range(len(d)):
        if i != j:
            if Methods.check_d(d, i, 1, j, 3):
                e = Methods.add_e(e, i, 1, j)
                e = Methods.add_e(e, j, 3, i)
            if Methods.check_d(d, i, 0, j, 2):
                e = Methods.add_e(e, i, 0, j)
                e = Methods.add_e(e, j, 2, i)
ttt = Methods.get_ttt(n, m)
loc = Methods.get_loc(d)


def sum_or_diff(fit_all, t, is_sum):
    global now_price, level
    if is_sum:
        now_price += fit_all[t]
        level += 1
    else:
        now_price -= fit_all[t]
        level -= 1


def dfs():
    global main_one, now_price, level
    if level == len(d):
        if now_price < main_one:
            main_one = now_price
        return
    for i in range(n):
        for j in range(m):
            if ttt[i][j] != -1:
                continue
            fit_all = Methods.get_fit(d)
            for k in range(4):
                fix = None
                if k == 0 and i != 0:
                    fix = i - 1, j
                elif k == 1 and j != m - 1:
                    fix = i, j + 1
                elif k == 2 and i != n - 1:
                    fix = i + 1, j
                elif k == 3 and j != 0:
                    fix = i, j - 1
                if fix is None:
                    continue
                if ttt[fix[0]][fix[1]] == -1:
                    continue
                super_set = ttt[fix[0]][fix[1]]
                relation = Methods.calculator(k)
                for neighbor in e[super_set][relation]:
                    if loc[neighbor] != (-1, -1):
                        continue
                    if fit_all[neighbor] is None:
                        fit_all[neighbor] = d[super_set][relation]
                    if fit_all[neighbor] > d[super_set][relation]:
                        fit_all[neighbor] = d[super_set][relation]
            if Methods.find(fit_all):
                num = len(fit_all)
                for t in range(num):
                    if fit_all[t]:
                        sum_or_diff(fit_all, t, True)
                        loc[t] = i, j
                        ttt[i][j] = t
                        if main_one > now_price:
                            dfs()
                        sum_or_diff(fit_all, t, False)
                        ttt[loc[t][0]][loc[t][1]] = -1
                        loc[t] = (-1, -1)


for i in range(len(d)):
    loc[i] = 0, 0
    ttt[0][0] = i
    level += 1
    dfs()
    if m * n == 12:
        print(main_one)
        break
    level -= 1
    ttt[loc[i][0]][loc[i][1]] = -1
    loc[i] = (-1, -1)
if m * n != 12:
    print(main_one)
