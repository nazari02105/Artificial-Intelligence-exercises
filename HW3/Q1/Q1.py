# https://www.geeksforgeeks.org/building-and-visualizing-sudoku-game-using-pygame/python-sudoku-solver/
# https://github.com/speix/sudoku-solver/blob/master/driver.py
# https://www.askpython.com/python/examples/sudoku-solver-in-python
# https://onestepcode.com/sudoku-solver-python/

row_names = 'ABCDEFGHI'
col_names = '123456789'
main_nodes = [a + b for a in row_names for b in col_names]


class Helper:
    @staticmethod
    def batch(main_list, batch_size):
        to_return = [[]]
        counter = 0
        index = 0
        for i in main_list:
            to_return[index].append(i)
            counter += 1
            if counter % batch_size == 0:
                index += 1
                to_return.append([])
        return to_return[:-1]

    @staticmethod
    def get_col(main_list):
        to_return = []
        for i in range(9):
            to_return.append([])
        counter = 0
        for i in main_list:
            to_return[counter % 9].append(i)
            counter += 1
        return to_return

    @staticmethod
    def get_each_node(main_list):
        to_return = []
        for i in range(9):
            to_return.append([])
        first_counter = 0
        second_counter = 0
        third_counter = 0
        forth_counter = 0
        for i in main_list:
            to_return[first_counter // 3 + third_counter * 3].append(i)
            first_counter += 1
            if first_counter % 9 == 0:
                first_counter = 0
                second_counter += 1
            if second_counter % 3 == 0 and second_counter != 0 and second_counter != forth_counter:
                forth_counter = second_counter
                third_counter += 1
        return to_return

    @staticmethod
    def get_relatives_of_each_node(main_list):
        to_return = {}
        for i in main_list:
            for j in i:
                if j in to_return.keys():
                    to_return[j].append(i)
                else:
                    the_list = [i]
                    to_return[j] = the_list
        return to_return

    @staticmethod
    def result(values):
        main_str = ""
        counter = 0
        each_row = ""
        for key, value in values.items():
            each_row += str(value) + " "
            counter += 1
            if counter % 9 == 0:
                each_row = each_row.strip() + "\n"
                main_str += each_row
                each_row = ""
        print(main_str)

    @staticmethod
    def get_unique_relatives_of_each_node(main_dict):
        to_return = {}
        for key in main_dict.keys():
            the_list = []
            for i in main_dict[key]:
                for j in i:
                    the_list.append(j)
            while key in the_list:
                the_list.remove(key)
            to_return[key] = set(the_list)
        return to_return

    @staticmethod
    def get_input():
        user_input = []
        for i in range(9):
            x = input().split()
            user_input += x

        user_input = ''.join(user_input)
        return user_input

    @staticmethod
    def get_choices(main_dict):
        for k in main_dict.keys():
            if len(main_dict[k]) != 1:
                neighbours = node_pairs[k]
                main_list = []
                for neighbour in neighbours:
                    if len(main_dict[neighbour]) == 1:
                        main_list.append(main_dict[neighbour])
                peer_values = set(main_list)
                main_dict[k] = ''.join(set(main_dict[k]) - peer_values)
        return main_dict

    @staticmethod
    def map_input(main_input, nodes):
        to_return = {}
        for i in range(81):
            to_return[nodes[i]] = main_input[i]
        return to_return

    @staticmethod
    def change_empty_nodes(main_dict):
        for k, v in main_dict.items():
            if v == '.':
                main_dict[k] = '123456789'
        return main_dict

    @staticmethod
    def make_possible_moves(grid_dict_in_possible_moves):
        does_change = False
        while not does_change:
            solved_values_before = Helper.number_of_solved(grid_dict_in_possible_moves)
            grid_dict_in_possible_moves = Helper.get_choices(grid_dict_in_possible_moves)
            grid_dict_in_possible_moves = Helper.consider_choices(grid_dict_in_possible_moves)
            solved_values_after = Helper.number_of_solved(grid_dict_in_possible_moves)
            does_change = solved_values_before == solved_values_after
            if Helper.has_answer(grid_dict_in_possible_moves):
                return False
        return grid_dict_in_possible_moves

    @staticmethod
    def easiest_to_solve(main_dict):
        main_list = []
        for key in main_dict.keys():
            if len(main_dict[key]) > 1:
                main_list.append((key, len(main_dict[key])))
        node_name, node_length = None, 10
        for pair in main_list:
            if pair[1] < node_length:
                node_name, node_length = pair
        return node_length, node_name

    @staticmethod
    def get_main_list():
        rows, cols, boxes = Helper.batch(main_nodes, 9), Helper.get_col(main_nodes), Helper.get_each_node(main_nodes)
        to_return = rows + cols + boxes
        return to_return

    @staticmethod
    def number_of_solved(main_dict):
        counter = 0
        for key in main_dict.keys():
            if len(main_dict[key]) == 1:
                counter += 1
        return counter

    @staticmethod
    def consider_choices(main_dict):
        for unit in all_parts_list:
            for num in '123456789':
                main_list = []
                for box in unit:
                    if num in main_dict[box]:
                        main_list.append(box)
                if len(main_list) == 1:
                    main_dict[main_list[0]] = num
        return main_dict

    @staticmethod
    def has_answer(main_dict):
        for key in main_dict.keys():
            if len(main_dict[key]) == 0:
                return True
        return False

    @staticmethod
    def is_solved(main_dict):
        main_len = len(main_dict.keys())
        counter = 0
        for key in main_dict.keys():
            if len(main_dict[key]) == 1:
                counter += 1
        return counter == main_len


def csp_solve(grid_dict_in_csp_solver):
    grid_dict_in_csp_solver = Helper.make_possible_moves(grid_dict_in_csp_solver)
    if grid_dict_in_csp_solver is False:
        return False
    if Helper.is_solved(grid_dict_in_csp_solver):
        return grid_dict_in_csp_solver
    node_length, node_name = Helper.easiest_to_solve(grid_dict_in_csp_solver)
    for digit in grid_dict_in_csp_solver[node_name]:
        new_sudoku = grid_dict_in_csp_solver.copy()
        new_sudoku[node_name] = digit
        try_to_solve_new_config = csp_solve(new_sudoku)
        if try_to_solve_new_config:
            return try_to_solve_new_config


all_parts_list = Helper.get_main_list()
node_pairs = Helper.get_unique_relatives_of_each_node(Helper.get_relatives_of_each_node(all_parts_list))
solved_sudoku = csp_solve(Helper.change_empty_nodes(Helper.map_input(Helper.get_input(), main_nodes)))
Helper.result(solved_sudoku)
