import numpy as np
import math
import copy

OLY_X, OLY_Y = -133, 18
ARS_X, ARS_Y = -121, -9
PAV_X, PAV_Y = -113, 1
ASC_X, ASC_Y = -104, 12
N = 1000

first_mountain = list()
second_mountain = list()
third_mountain = list()
forth_mountain = list()


class Node:
    def __init__(self, class_x, class_y, weigh):
        self.class_x = class_x
        self.class_y = class_y
        self.weigh = weigh

    def __str__(self):
        return str(self.class_x) + "&" + str(self.class_y) + "&" + str(self.weigh)


class Helper:
    @staticmethod
    def get_input(name):
        global first_mountain, second_mountain, third_mountain, forth_mountain
        input()
        for _ in range(20):
            number = float(input())
            if name == "first":
                first_mountain.append(number)
            elif name == "second":
                second_mountain.append(number)
            elif name == "third":
                third_mountain.append(number)
            else:
                forth_mountain.append(number)

    @staticmethod
    def get_uniform_list(start, end):
        global N
        return np.random.uniform(start, end, N)

    @staticmethod
    def get_normal_list(mean, var):
        global N
        return np.random.normal(mean, var, N)

    @staticmethod
    def concat_arrays(first, second):
        to_return = list()
        for index in range(len(first)):
            obj = Node(first[index], second[index], 0.001)
            to_return.append(obj)
        return to_return

    @staticmethod
    def do_round(nodes_in_function):
        global first_distribution, second_distribution
        to_return = copy.deepcopy(nodes_in_function)
        for index in range(N):
            obj = Node(
                nodes_in_function[index].class_x + first_distribution[index],
                nodes_in_function[index].class_y + second_distribution[index],
                nodes_in_function[index].weigh
            )
            to_return[index] = obj
        return to_return

    @staticmethod
    def calculate_distance(first_x, first_y, second_x, second_y):
        return np.sqrt((math.pow(first_x - second_x, 2)) + (math.pow(first_y - second_y, 2)))

    @staticmethod
    def calculate_pdf(class_x, expected, std):
        return np.exp(-1 * ((class_x - expected) / std) * ((class_x - expected) / std) / 2) * (
                1 / (std * np.sqrt(2 * 3.1415926535)))

    @staticmethod
    def each_mountain(node, name, index):
        if name == "first":
            returned_dist = Helper.calculate_distance(node.class_x, node.class_y, OLY_X, OLY_Y)
            returned_weigh = Helper.calculate_pdf(returned_dist - first_mountain[index], 2, 1)
        elif name == "second":
            returned_dist = Helper.calculate_distance(node.class_x, node.class_y, ARS_X, ARS_Y)
            returned_weigh = Helper.calculate_pdf(returned_dist - second_mountain[index], 2, 1)
        elif name == "third":
            returned_dist = Helper.calculate_distance(node.class_x, node.class_y, ASC_X, ASC_Y)
            returned_weigh = Helper.calculate_pdf(returned_dist - third_mountain[index], 2, 1)
        else:
            returned_dist = Helper.calculate_distance(node.class_x, node.class_y, PAV_X, PAV_Y)
            returned_weigh = Helper.calculate_pdf(returned_dist - forth_mountain[index], 2, 1)
        return returned_weigh

    @staticmethod
    def change_weigh(nodes_in_function, main_index):
        global N
        to_return = copy.deepcopy(nodes_in_function)
        for index in range(N):
            weigh1 = Helper.each_mountain(to_return[index], "first", main_index)
            weigh2 = Helper.each_mountain(to_return[index], "second", main_index)
            weigh3 = Helper.each_mountain(to_return[index], "third", main_index)
            weigh4 = Helper.each_mountain(to_return[index], "forth", main_index)
            (to_return[index]).weigh = weigh1 + weigh2 + weigh3 + weigh4
        return to_return

    @staticmethod
    def do_sum(nodes_in_function):
        w_sum = sum_x = sum_y = 0
        for index in range(N):
            w_sum += (nodes_in_function[index]).weigh
            sum_x += (nodes_in_function[index]).class_x * (nodes_in_function[index]).weigh
            sum_y += (nodes_in_function[index]).class_y * (nodes_in_function[index]).weigh
        return sum_x / w_sum, sum_y / w_sum


Helper.get_input("first")
Helper.get_input("second")
Helper.get_input("third")
Helper.get_input("forth")

# because mountains x are in range of (-160, 100)
x = Helper.get_uniform_list(-160, -100)
# because mountains y are in range of (-20, 40)
y = Helper.get_uniform_list(-20, 40)
nodes = Helper.concat_arrays(x, y)
for i in range(len(first_mountain)):
    first_distribution = Helper.get_normal_list(2, 1)
    second_distribution = Helper.get_normal_list(1, 1)
    nodes = Helper.do_round(nodes)
    nodes = Helper.change_weigh(nodes, i)
    nodes.sort(key=lambda main_key: main_key.weigh, reverse=True)
    nodes[500:] = nodes[:500]
x_mean, y_mean = Helper.do_sum(nodes)
print(int(np.ceil(x_mean / 10) * 10))
print(int(np.ceil(y_mean / 10) * 10))
