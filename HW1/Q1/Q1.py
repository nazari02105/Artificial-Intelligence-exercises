# I used these sites to solve this problem:
# https://www.geeksforgeeks.org/dijkstras-shortest-path-algorithm-greedy-algo-7/
# https://www.programiz.com/dsa/dijkstra-algorithm
# https://www.geeksforgeeks.org/shortest-path-weighted-graph-weight-edge-1-2/
import math
import gc


def does_exists(my_dict, key):
    return key not in my_dict["vertices"].keys()


def get_min(my_dict):
    to_return = -1
    main_max = math.inf
    for key in my_dict.keys():
        if my_dict[key] < main_max:
            to_return = key
            main_max = my_dict[key]
    return to_return


def delete_key(my_dict, key):
    del my_dict[key]
    return my_dict


def check_availability_of_car(vertices, current, length_from):
    if vertices[current][-1]:
        length_from /= 2
    return length_from


def print_result(my_list):
    result = ""
    for i in my_list:
        result += str(i) + " "
    result = result.strip()
    print(result)


def pure_node():
    return [{}, math.inf, math.inf, False, False, None, None, False]


k = int(input())
for i in range(k):
    main_graph = {"vertices": {}}
    vertices = {}
    number_of_cities, number_of_roads = [int(input_number) for input_number in input().split()]
    for m in range(number_of_roads):
        start, finish, distance = [int(input_number) for input_number in input().split()]
        if does_exists(main_graph, start):
            vertices[start] = pure_node()
            main_graph["vertices"][start] = None
        if does_exists(main_graph, finish):
            vertices[finish] = pure_node()
            main_graph["vertices"][finish] = None
        vertices[start][0][finish] = distance
        vertices[finish][0][start] = distance
    t = int(input())
    all_lengths = {}
    thief_cities = [int(input_number) for input_number in input().split()]
    for s in thief_cities:
        vertices[s][2] = 0
        all_lengths[s] = 0
    c = int(input())
    car_cities = [int(input_number) for input_number in input().split()]
    for c in car_cities:
        vertices[c][-1] = True
    source, target = [int(input_number) for input_number in input().split()]
    while len(all_lengths.keys()) != 0:
        current_source_node = get_min(all_lengths)
        vertices[current_source_node][4] = True
        all_lengths = delete_key(all_lengths, current_source_node)
        for node_dist in vertices[current_source_node][0].keys():
            if vertices[node_dist][4]:
                continue
            neighbour = node_dist
            length_from = vertices[current_source_node][0][node_dist]
            length_from = check_availability_of_car(vertices, current_source_node, length_from)
            if vertices[node_dist][2] > vertices[current_source_node][2] + length_from:
                vertices[node_dist][2] = vertices[current_source_node][2] + length_from
                vertices[node_dist][6] = current_source_node
                all_lengths[neighbour] = vertices[node_dist][2]
                if vertices[current_source_node][-1]:
                    vertices[node_dist][-1] = True
    can_tintin = False
    vertices[source][1] = 0
    all_lengths = dict()
    all_lengths[source] = 0
    while len(all_lengths.keys()) != 0:
        current_source_node = get_min(all_lengths)
        vertices[current_source_node][3] = True
        all_lengths = delete_key(all_lengths, current_source_node)
        if current_source_node == target and vertices[current_source_node][5] is not None:
            can_tintin = True
        for node_dist in vertices[current_source_node][0].keys():
            if vertices[node_dist][3]:
                continue
            neighbour = node_dist
            length_from = vertices[current_source_node][0][node_dist]
            if vertices[current_source_node][1] + length_from > vertices[node_dist][2]:
                continue
            if vertices[node_dist][1] > vertices[current_source_node][1] + length_from:
                vertices[node_dist][1] = vertices[current_source_node][1] + length_from
                vertices[node_dist][5] = current_source_node
                all_lengths[neighbour] = vertices[node_dist][1]
    gc.collect()
    if can_tintin:
        print(vertices[target][1])
        path = list()
        path.append(target)
        while vertices[target][5] and target != source:
            path.append(vertices[target][5])
            target = vertices[target][5]
        print(len(path))
        b = path[::-1]
        print_result(b)
    else:
        print("Poor Tintin")
