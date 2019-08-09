from utils import *
import collections


row_units = [cross(r, cols) for r in rows]
column_units = [cross(rows, c) for c in cols]
square_units = [cross(rs, cs) for rs in ('ABC','DEF','GHI') for cs in ('123','456','789')]
unitlist = row_units + column_units + square_units

# TODO: Update the unit list to add the new diagonal units
unitlist = unitlist


# Must be called after all units (including diagonals) are added to the unitlist
units = extract_units(unitlist, boxes)
peers = extract_peers(units, boxes)




################## custom functions ########################
################## custom functions ########################
def check_if_solved(values):
    ## check if sudoku is solved
    if all(len(values[key]) == 1 for key in boxes):
        return values # sudoku is solved
    else:
        False
def local_box(key,values):
    letters_cat={'A':'ABC','B':'DEF','C':'GHI'}
    numbers_cat={'1':'123','2':'456','3':'789'}
    letters_key =[dict_key for dict_key,value in letters_cat.items() if key[0] in value]
    numbers_key =[dict_key for dict_key,value in numbers_cat.items() if key[1] in value]
    return cross(letters_cat[letters_key[0]],numbers_cat[numbers_key[0]])
def remove_value_from(group_keys,numbers_to_remove,values,exception=[]):
    '''function to remove number from sudoku library'''
    ## loop over each number
    for number in numbers_to_remove:
        ## loop over the dictionary of group we are interested in
        for key in group_keys:
            # check that the field is not already completed
            if len(values[key])>1 and key not in exception:
                # remove value from the string by replacing it with ""
                values[key]=values[key].replace(number,'')
    return values
def local_row_column(key,values):
    '''function to get row of a key 
    :key: String box_name eg 'A1','A2' 
    :values: Dict Sudoku board'''
    return [box_name for box_name in values.keys() if ((key[0] in box_name) or (key[1] in box_name))]
def local_row(key,values):
    '''function to get row of a key 
    :key: String box_name eg 'A1','A2' 
    :values: Dict Sudoku board'''
    return [box_name for box_name in values.keys() if ((key[0] in box_name))]
def local_column(key,values):
    '''function to get row of a key 
    :key: String box_name eg 'A1','A2' 
    :values: Dict Sudoku board'''
    return [box_name for box_name in values.keys() if ((key[1] in box_name))]
def local_group(key,values):
    '''function to get row,column,and local box of a key 
    :key: String box_name eg 'A1','A2' 
    :values: Dict Sudoku board
    :return: list of local group key
    '''
    key_local_box= local_box(key,values)
#     print(key_local_box)
    return [box_name for box_name in values.keys() if ((key[0] in box_name) or (key[1] in box_name)) or (box_name in key_local_box)]
# def elimination(values):
#     new_sudoku = values.copy()
#     stall=False
#     while stall==False:
#         original_sudoku= new_sudoku.copy()
#         for key in new_sudoku.keys():
#             if len(new_sudoku[key])==1:
#                 remove_value_from(local_group(key,new_sudoku),new_sudoku[key] ,new_sudoku ,exception=[key])
#         if original_sudoku == new_sudoku:
#             stall=True
#     return new_sudoku
        
def check_if_in_local_group(values,key,value_to_check):
    for local_group_key in local_group(key,values):
        if (key != local_group_key) and (str(value_to_check) in values[local_group_key]):
            return True
    return False
        
# def only_choice(values):
#     new_sudoku = values.copy()
#     stall=False
#     while stall==False:
#         original_sudoku= new_sudoku.copy()
#         for key in new_sudoku.keys():
#             #check if box not complete and list local_group keys
#             if len(new_sudoku[key])>1:
#                 for potential_value in new_sudoku[key]:
#                     if not check_if_in_local_group(values,key,potential_value):
#                         new_sudoku[key]=potential_value
#         if original_sudoku == new_sudoku:
#             stall=True
#     return new_sudoku
# def naked_twins(values):
#     new_sudoku = values.copy()
#     stall=False
#     while stall==False:
#         original_sudoku= new_sudoku.copy()
#         ## check rows
#         for unit in unitlist:
# #             print(f'unit is {unit}')
#             two_value_dict = {key:new_sudoku[key] for key in unit if len(new_sudoku[key])==2}
#             #check for duplicate for each key
#             duplicate_values=[item for item, count in collections.Counter(two_value_dict.values()).items() if count > 1]
#             if duplicate_values:
#                 for duplicate_value in duplicate_values:
#                     copy_new_sudoku=new_sudoku.copy()
#                     new_sudoku=remove_value_from(unit,duplicate_value,new_sudoku,exception=[key for key in unit if new_sudoku[key]==duplicate_value])
#                     if copy_new_sudoku != new_sudoku:
#                         print(f'found duplicate values of {duplicate_values}')
#                         print(f'dict_before and after removing duplicats')
#                         display(copy_new_sudoku)
#                         display(new_sudoku)
#         if original_sudoku == new_sudoku:
#             stall=True
#         print('loop finished')
#     return new_sudoku

# def reduce_puzzle(values):
#     new_sudoku = values.copy()
#     stall=False
#     while stall==False:
#         original_sudoku= new_sudoku.copy()
#         new_sudoku=elimination(new_sudoku)
#         new_sudoku=only_choice(new_sudoku)
#         new_sudoku=naked_twins(new_sudoku)
#         if original_sudoku == new_sudoku:
#             stall=True
#     return new_sudoku
# def search(values):
#     "Using depth-first search and propagation, try all possible values."
#     # First, reduce the puzzle using the previous function
#     values = reduce_puzzle(values)
#     if values is False:
#         return False ## Failed earlier
#     if all(len(values[s]) == 1 for s in boxes): 
#         return values ## Solved!
#     # Choose one of the unfilled squares with the fewest possibilities
#     n,s = min((len(values[s]), s) for s in boxes if len(values[s]) > 1)
#     # Now use recurrence to solve each one of the resulting sudokus, and 
#     for value in values[s]:
#         new_sudoku = values.copy()
#         new_sudoku[s] = value
#         attempt = search(new_sudoku)
#         if attempt:
#             return attempt
        
###### compulsary functions#####################


def naked_twins(values):
    """Eliminate values using the naked twins strategy.

    The naked twins strategy says that if you have two or more unallocated boxes
    in a unit and there are only two digits that can go in those two boxes, then
    those two digits can be eliminated from the possible assignments of all other
    boxes in the same unit.

    Parameters
    ----------
    values(dict)
        a dictionary of the form {'box_name': '123456789', ...}

    Returns
    -------
    dict
        The values dictionary with the naked twins eliminated from peers

    Notes
    -----
    Your solution can either process all pairs of naked twins from the input once,
    or it can continue processing pairs of naked twins until there are no such
    pairs remaining -- the project assistant test suite will accept either
    convention. However, it will not accept code that does not process all pairs
    of naked twins from the original input. (For example, if you start processing
    pairs of twins and eliminate another pair of twins before the second pair
    is processed then your code will fail the PA test suite.)

    The first convention is preferred for consistency with the other strategies,
    and because it is simpler (since the reduce_puzzle function already calls this
    strategy repeatedly).

    See Also
    --------
    Pseudocode for this algorithm on github:
    https://github.com/udacity/artificial-intelligence/blob/master/Projects/1_Sudoku/pseudocode.md
    """
    new_sudoku = values.copy()
    stall=False
    while stall==False:
        original_sudoku= new_sudoku.copy()
        ## check rows
        for unit in unitlist:
#             print(f'unit is {unit}')
            two_value_dict = {key:new_sudoku[key] for key in unit if len(new_sudoku[key])==2}
            #check for duplicate for each key
            duplicate_values=[item for item, count in collections.Counter(two_value_dict.values()).items() if count > 1]
            if duplicate_values:
                for duplicate_value in duplicate_values:
                    copy_new_sudoku=new_sudoku.copy()
                    new_sudoku=remove_value_from(unit,duplicate_value,new_sudoku,exception=[key for key in unit if new_sudoku[key]==duplicate_value])
                    if copy_new_sudoku != new_sudoku:
                        print(f'found duplicate values of {duplicate_values}')
                        print(f'dict_before and after removing duplicats')
                        display(copy_new_sudoku)
                        display(new_sudoku)
        if original_sudoku == new_sudoku:
            stall=True
        print('loop finished')
    return new_sudoku
    

def eliminate(values):
    """Apply the eliminate strategy to a Sudoku puzzle

    The eliminate strategy says that if a box has a value assigned, then none
    of the peers of that box can have the same value.

    Parameters
    ----------
    values(dict)
        a dictionary of the form {'box_name': '123456789', ...}

    Returns
    -------
    dict
        The values dictionary with the assigned values eliminated from peers
    """
    new_sudoku = values.copy()
    stall=False
    while stall==False:
        original_sudoku= new_sudoku.copy()
        for key in new_sudoku.keys():
            if len(new_sudoku[key])==1:
                remove_value_from(local_group(key,new_sudoku),new_sudoku[key] ,new_sudoku ,exception=[key])
        if original_sudoku == new_sudoku:
            stall=True
    return new_sudoku


def only_choice(values):
    """Apply the only choice strategy to a Sudoku puzzle

    The only choice strategy says that if only one box in a unit allows a certain
    digit, then that box must be assigned that digit.

    Parameters
    ----------
    values(dict)
        a dictionary of the form {'box_name': '123456789', ...}

    Returns
    -------
    dict
        The values dictionary with all single-valued boxes assigned

    Notes
    -----
    You should be able to complete this function by copying your code from the classroom
    """
    new_sudoku = values.copy()
    stall=False
    while stall==False:
        original_sudoku= new_sudoku.copy()
        for key in new_sudoku.keys():
            #check if box not complete and list local_group keys
            if len(new_sudoku[key])>1:
                for potential_value in new_sudoku[key]:
                    if not check_if_in_local_group(values,key,potential_value):
                        new_sudoku[key]=potential_value
        if original_sudoku == new_sudoku:
            stall=True
    return new_sudoku


def reduce_puzzle(values):
    """Reduce a Sudoku puzzle by repeatedly applying all constraint strategies

    Parameters
    ----------
    values(dict)
        a dictionary of the form {'box_name': '123456789', ...}

    Returns
    -------
    dict or False
        The values dictionary after continued application of the constraint strategies
        no longer produces any changes, or False if the puzzle is unsolvable 
    """
    new_sudoku = values.copy()
    stall=False
    while stall==False:
        original_sudoku= new_sudoku.copy()
        new_sudoku=eliminate(new_sudoku)
        new_sudoku=only_choice(new_sudoku)
        new_sudoku=naked_twins(new_sudoku)
        if original_sudoku == new_sudoku:
            stall=True
    return new_sudoku


def search(values):
    """Apply depth first search to solve Sudoku puzzles in order to solve puzzles
    that cannot be solved by repeated reduction alone.

    Parameters
    ----------
    values(dict)
        a dictionary of the form {'box_name': '123456789', ...}

    Returns
    -------
    dict or False
        The values dictionary with all boxes assigned or False

    Notes
    -----
    You should be able to complete this function by copying your code from the classroom
    and extending it to call the naked twins strategy.
    """
    values = reduce_puzzle(values)
    if values is False:
        return False ## Failed earlier
    if all(len(values[s]) == 1 for s in boxes): 
        return values ## Solved!
    # Choose one of the unfilled squares with the fewest possibilities
    n,s = min((len(values[s]), s) for s in boxes if len(values[s]) > 1)
    # Now use recurrence to solve each one of the resulting sudokus, and 
    for value in values[s]:
        new_sudoku = values.copy()
        new_sudoku[s] = value
        attempt = search(new_sudoku)
        if attempt:
            return attempt




def solve(grid):
    """Find the solution to a Sudoku puzzle using search and constraint propagation

    Parameters
    ----------
    grid(string)
        a string representing a sudoku grid.
        
        Ex. '2.............62....1....7...6..8...3...9...7...6..4...4....8....52.............3'

    Returns
    -------
    dict or False
        The dictionary representation of the final sudoku grid or False if no solution exists.
    """
    values = grid2values(grid)
    values = search(values)
    return values


if __name__ == "__main__":
    diag_sudoku_grid = '2.............62....1....7...6..8...3...9...7...6..4...4....8....52.............3'
    display(grid2values(diag_sudoku_grid))
    result = solve(diag_sudoku_grid)
    display(result)

    try:
        import PySudoku
        PySudoku.play(grid2values(diag_sudoku_grid), result, history)

    except SystemExit:
        pass
    except:
        print('We could not visualize your board due to a pygame issue. Not a problem! It is not a requirement.')
