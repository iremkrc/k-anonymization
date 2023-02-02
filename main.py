##############################################################################
# The skeleton of code was created by Efehan Guner  (efehanguner21@ku.edu.tr)#
# Note: requires Python 3.5+                                                 #
##############################################################################

import csv
from enum import unique
import glob
import itertools
import os
import sys
from copy import deepcopy
import numpy as np
import datetime

if sys.version_info[0] < 3 or sys.version_info[1] < 5:
    sys.stdout.write("Requires Python 3.x.\n")
    sys.exit(1)


# This class represents a node in the DGH. It has a name, a level, a list of children and a parent.
class Node(object):
    def __init__(self, name: str, level: int, parent=None):
        self.name = name
        self.level = level
        self.children = []
        self.parent = parent

    def __str__(self):
        return "Node: " + self.name

    def add_child(self, obj):
        self.children.append(obj)

# Returns true if the given dataset satisfies k-anonymity, false otherwise. 
# Checks how many identical lines are there for each line. 
# If there are other k-1 identical lines for each line, it returns true; otherwise returns false
def check_k_anon(dataset, k, sensitive) -> bool:
    dataset_copy = []
    for person in dataset:
        person_copy = person.copy()
        dataset_copy.append(person_copy)

    for p in dataset_copy:
        for s in sensitive:
            p.pop(s)

    for p in dataset_copy:
        if dataset_copy.count(p) < k:
            return False
    return True


# Counts the number of nodes that are leaves of the given node.
def find_leaf_number(node: Node) -> int:
    ret = 0
    # if children is empty, the node is a leaf!
    if len(node.children) == 0:
        return 1
    else:
        for child in node.children:
            ret += find_leaf_number(child)
        return ret


def lowest_common_ancestor(node1: Node, node2: Node) -> Node:
    if(node1.level == node2.level):
        if(node1.name == node2.name):
            return node1
        else:
            return lowest_common_ancestor(node1.parent, node2.parent)
    elif(node1.level > node2.level):
        return lowest_common_ancestor(node1.parent, node2)
    else:
        return lowest_common_ancestor(node1, node2.parent)


# Returns the lowest common ancestor of the elements in the given list of nodes.
def lowest_common_ancestor_list(nodes) -> Node:
    assert len(nodes) != 0, "list should not be empty"
    ancestor = lowest_common_ancestor(nodes[0], nodes[0])
    for i in range(1, len(nodes)):
        ancestor = lowest_common_ancestor(ancestor, nodes[i])
    return ancestor


# Returns a dictionary whose keys are attribute names and whose values are the first line of the attribute's DGH.
def first_elements_DGH(DGH_folder: str):
    first_elements = {}
    DGHs = read_DGHs(DGH_folder)

    for DGH_file in glob.glob(DGH_folder + "/*.txt"):
        attribute_name = os.path.basename(DGH_file)[:-4]
        with open(DGH_file) as f:
            first_line = f.readline()
        first_line = "".join(first_line.split())
        first_elements[attribute_name] = DGHs[attribute_name][first_line]
    return first_elements


# Returns a dictionary whose keys are attribute names and whose values are the maximum levels that a node can have in this attribute.
def max_depths_DGH(DGHs):
    max_depths = {}
    for att in list(DGHs.keys()):
        max_level = 0
        for val in list(DGHs[att].keys()):
            node = DGHs[att][val]
            if node.level > max_level:
                max_level = node.level
        max_depths[att] = max_level
    return max_depths


def not_visited_number(arr) -> int:
    counter = 0
    for i in range(len(arr)):
        if arr[i] == 0:
            counter += 1
    return counter


def calculate_dist(node1: Node, node2: Node, first: Node):
    lca = lowest_common_ancestor(node1, node2)
    raw1 = (find_leaf_number(node1) - 1) / (find_leaf_number(first) - 1)
    anon1 = (find_leaf_number(lca) - 1) / (find_leaf_number(first) - 1)
    cost_LM1 = abs(raw1 - anon1)
    
    raw2 = (find_leaf_number(node2) - 1) / (find_leaf_number(first) - 1)
    anon2 = (find_leaf_number(lca) - 1) / (find_leaf_number(first) - 1)
    cost_LM2 = abs(raw2 - anon2)
    return cost_LM1 + cost_LM2


def read_dataset(dataset_file: str):
    """ Read a dataset into a list and return.

    Args:
        dataset_file (str): path to the dataset file.

    Returns:
        list[dict]: a list of dataset rows.
    """
    result = []
    with open(dataset_file) as f:
        records = csv.DictReader(f)
        for row in records:
            result.append(row)
    return result


def write_dataset(dataset, dataset_file: str) -> bool:
    """ Writes a dataset to a csv file.

    Args:
        dataset: the data in list[dict] format
        dataset_file: str, the path to the csv file

    Returns:
        bool: True if succeeds.
    """
    assert len(dataset)>0, "The anonymized dataset is empty."
    keys = dataset[0].keys()
    with open(dataset_file, 'w', newline='')  as output_file:
        dict_writer = csv.DictWriter(output_file, keys)
        dict_writer.writeheader()
        dict_writer.writerows(dataset)
    return True


def read_DGH(DGH_file: str):
    """ Reads one DGH file and returns in desired format.

    Args:
        DGH_file (str): the path to DGH file.
    """
    # returns a dict where key is the name of attiribute (eg. [0-50]), value is a node of that attribute.
    dgh = {}
    f = open(DGH_file, 'r')

    prev_node = None
    
    for row in f.readlines():
        name = "".join(row.split())
        level = row.count('\t')
        node = Node(name, level)
        
        if level != 0:
            if level == prev_node.level:
                node.parent = prev_node.parent
            elif level == prev_node.level + 1:
                node.parent = prev_node
            else:
                prev_parent = prev_node.parent
                while prev_parent.level != level:
                    prev_parent = prev_parent.parent
                node.parent = prev_parent.parent
            node.parent.add_child(node)
            
        dgh[name] = node
        prev_node = node

    return dgh


def read_DGHs(DGH_folder: str) -> dict:
    """ Read all DGH files from a directory and put them into a dictionary.

    Args:
        DGH_folder (str): the path to the directory containing DGH files.

    Returns:
        dict: a dictionary where each key is attribute name and values
            are DGHs in your desired format.
    """
    DGHs = {}
    for DGH_file in glob.glob(DGH_folder + "/*.txt"):
        attribute_name = os.path.basename(DGH_file)[:-4]
        DGHs[attribute_name] = read_DGH(DGH_file)

    return DGHs


# Same with cost_LM function but takes the datasets as lists.
def cost_LM_files(raw_dataset, anonymized_dataset, DGH_folder) -> float:
    assert(len(raw_dataset)>0 and len(raw_dataset) == len(anonymized_dataset)
        and len(raw_dataset[0]) == len(anonymized_dataset[0]))
    DGHs = read_DGHs(DGH_folder)

    att_list = list(DGHs.keys())
    firsts = first_elements_DGH(DGH_folder)
    cost = 0
    quasi_atts = []
    # e.g: ['age', 'workclass', 'education', 'income' ... ]
    person_att_list = list(raw_dataset[0].keys())
    for j in person_att_list:
        if j in att_list:
            quasi_atts.append(j)
    weight = len(quasi_atts)
    for i in range(len(raw_dataset)):
        for j in quasi_atts:
            anon_att_name = anonymized_dataset[i][j]
            anon_LM = (find_leaf_number(DGHs[j][anon_att_name]) - 1) / (find_leaf_number(firsts[j]) - 1)
            cost += anon_LM * 1/weight
    return cost    


def cost_MD(raw_dataset_file: str, anonymized_dataset_file: str,
    DGH_folder: str) -> float:
    """Calculate Distortion Metric (MD) cost between two datasets.

    Args:
        raw_dataset_file (str): the path to the raw dataset file.
        anonymized_dataset_file (str): the path to the anonymized dataset file.
        DGH_folder (str): the path to the DGH directory.

    Returns:
        float: the calculated cost.
    """
    raw_dataset = read_dataset(raw_dataset_file)
    anonymized_dataset = read_dataset(anonymized_dataset_file)
    assert(len(raw_dataset)>0 and len(raw_dataset) == len(anonymized_dataset)
        and len(raw_dataset[0]) == len(anonymized_dataset[0]))
    DGHs = read_DGHs(DGH_folder)

    att_list = list(DGHs.keys())
    cost = 0
    for i in range(len(raw_dataset)):
        # e.g: ['age', 'workclass', 'education', 'income' ... ]
        person_att_list = list(raw_dataset[i].keys())
        for j in person_att_list:
            if j in att_list:
                # e.g: 17
                raw_att_name = raw_dataset[i][j]
                # e.g: [10,20)
                anon_att_name = anonymized_dataset[i][j]
                raw_level = DGHs[j][raw_att_name].level
                anon_level = DGHs[j][anon_att_name].level
                cost += abs(raw_level - anon_level)

    return cost



def cost_LM(raw_dataset_file: str, anonymized_dataset_file: str,
    DGH_folder: str) -> float:
    """Calculate Loss Metric (LM) cost between two datasets.

    Args:
        raw_dataset_file (str): the path to the raw dataset file.
        anonymized_dataset_file (str): the path to the anonymized dataset file.
        DGH_folder (str): the path to the DGH directory.

    Returns:
        float: the calculated cost.
    """
    raw_dataset = read_dataset(raw_dataset_file) # returns list of dicts
    anonymized_dataset = read_dataset(anonymized_dataset_file)
    assert(len(raw_dataset)>0 and len(raw_dataset) == len(anonymized_dataset)
        and len(raw_dataset[0]) == len(anonymized_dataset[0]))
    DGHs = read_DGHs(DGH_folder)

    att_list = list(DGHs.keys())
    firsts = first_elements_DGH(DGH_folder)
    cost = 0
    quasi_atts = []
    # e.g: ['age', 'workclass', 'education', 'income' ... ]
    person_att_list = list(raw_dataset[0].keys())
    for j in person_att_list:
        if j in att_list:
            quasi_atts.append(j)
    weight = len(quasi_atts)
    for i in range(len(raw_dataset)):
        for j in quasi_atts:
            anon_att_name = anonymized_dataset[i][j]
            anon_LM = (find_leaf_number(DGHs[j][anon_att_name]) - 1) / (find_leaf_number(firsts[j]) - 1)
            cost += anon_LM * 1/weight
    return cost           
            

def random_anonymizer(raw_dataset_file: str, DGH_folder: str, k: int,
    output_file: str, s: int):
    """ K-anonymization a dataset, given a set of DGHs and a k-anonymity param.

    Args:
        raw_dataset_file (str): the path to the raw dataset file.
        DGH_folder (str): the path to the DGH directory.
        k (int): k-anonymity parameter.
        output_file (str): the path to the output dataset file.
        s (int): seed of the randomization function
    """
    raw_dataset = read_dataset(raw_dataset_file)
    DGHs = read_DGHs(DGH_folder)    

    for i in range(len(raw_dataset)): ##set indexing to not lose original places of records
        raw_dataset[i]['index'] = i

    raw_dataset = np.array(raw_dataset)
    np.random.seed(s) ## to ensure consistency between runs
    np.random.shuffle(raw_dataset)  ##shuffle the dataset to randomize

    clusters = []

    D = len(raw_dataset)
     
    count = D
    for i in range(D//k):
        if i==(D//k)-1:
            clusters.append(raw_dataset[D-count:])
        else:
            clusters.append(raw_dataset[D-count:D-count+k])
        count -= k
    
    person_att_list = list(raw_dataset[0].keys())
    att_list = list(DGHs.keys())
    att_dict = {}
    att_anon_dict = {}

    for cluster in clusters:
        for att in person_att_list:
            if att in att_list:
                att_dict[att] = []
                att_anon_dict[att] = []
        for person in cluster:
            for att in list(att_dict.keys()):
                att_dict[att].append(person[att])
        for att in list(att_dict.keys()):
            att_val_list = att_dict[att]
            for i in range(len(att_val_list)):
                att_val_list[i] = DGHs[att][att_val_list[i]]
            att_anon_dict[att] = lowest_common_ancestor_list(att_val_list)
        for person in cluster:
            for att in list(att_anon_dict.keys()):
                person[att] = att_anon_dict[att].name


    # END OF STUDENT'S CODE. Do not modify code in this function below this line.

    anonymized_dataset = [None] * D

    for cluster in clusters:        #restructure according to previous indexes
        for item in cluster:
            anonymized_dataset[item['index']] = item
            del item['index']

    write_dataset(anonymized_dataset, output_file)



def clustering_anonymizer(raw_dataset_file: str, DGH_folder: str, k: int,
    output_file: str):
    """ Clustering-based anonymization a dataset, given a set of DGHs.

    Args:
        raw_dataset_file (str): the path to the raw dataset file.
        DGH_folder (str): the path to the DGH directory.
        k (int): k-anonymity parameter.
        output_file (str): the path to the output dataset file.
    """
    raw_dataset = read_dataset(raw_dataset_file)
    DGHs = read_DGHs(DGH_folder)

    firsts = first_elements_DGH(DGH_folder)
    att_list = list(DGHs.keys())

    visited = [0] * len(raw_dataset)
    index = 0
    clusters = []
    while not_visited_number(visited) >= 2 * k: 
        tuple_list = []
        cluster = []
        if visited[index] == 0:
            visited[index] = 1
            cluster.append(raw_dataset[index])
            person_att_list = list(raw_dataset[index].keys())
            for j in range(index+1, len(raw_dataset)):
                if visited[j] == 0:
                    dist = 0
                    for i in person_att_list:
                        if i in att_list:
                            first = firsts[i]
                            dist += calculate_dist(DGHs[i][raw_dataset[index][i]], DGHs[i][raw_dataset[j][i]], first)
                    tuple_list.append((dist, j))
            tuple_list = sorted(tuple_list)
            tuple_list = tuple_list[:(k-1)]
            for t in tuple_list:
                visited[t[1]] = 1
                cluster.append(raw_dataset[t[1]])
            clusters.append(cluster)
        index += 1
    last_cluster  = []
    for i in range(len(visited)):
        if visited[i] == 0:
            last_cluster.append(raw_dataset[i])
    if len(last_cluster) != 0:
        clusters.append(last_cluster)

    person_att_list = list(raw_dataset[0].keys())
    att_dict = {}
    att_anon_dict = {}
 
    for cluster in clusters:
        for att in person_att_list:
            if att in att_list:
                att_dict[att] = []
                att_anon_dict[att] = []
        for person in cluster:
            for att in list(att_dict.keys()):
                att_dict[att].append(person[att])
        att_dict_key_list = list(att_dict.keys())
        for att in range(len(att_dict_key_list)):
            att_val_list = att_dict[att_dict_key_list[att]]
            for i in range(len(att_val_list)):
                att_val_list[i] = DGHs[att_dict_key_list[att]][att_val_list[i]]
            att_anon_dict[att_dict_key_list[att]] = lowest_common_ancestor_list(att_val_list)
        for person in cluster:
            for att in list(att_anon_dict.keys()):
                person[att] = att_anon_dict[att].name

    
    # Finally, write dataset to a file
    write_dataset(raw_dataset, output_file)



def bottomup_anonymizer(raw_dataset_file: str, DGH_folder: str, k: int,
    output_file: str):
    """ Bottom up-based anonymization a dataset, given a set of DGHs.

    Args:
        raw_dataset_file (str): the path to the raw dataset file.
        DGH_folder (str): the path to the DGH directory.
        k (int): k-anonymity parameter.
        output_file (str): the path to the output dataset file.
    """

    raw_dataset = read_dataset(raw_dataset_file)
    DGHs = read_DGHs(DGH_folder)

    max_depths = max_depths_DGH(DGHs)
    sample_person = raw_dataset[0]
    person_att_list = list(sample_person.keys())
    att_list = list(DGHs.keys())
    quasi_atts = []
    sensitive_atts = []
    for att in person_att_list:
        if att in att_list:
            quasi_atts.append(att)
        else:
            sensitive_atts.append(att)
    
    levels = []
    for att in quasi_atts:
        levels.append(max_depths[att])
    
    levels_arg = list(map(lambda x: range(x+1), levels))
    comb = list(itertools.product(*levels_arg))
    comb = sorted(comb, key = lambda x: sum(x))

    results = []
    result_lattice = float('inf')
    for t in comb:
        lattice = sum(t)
        if lattice > result_lattice:
            break
        anon_dataset = []
        for p in raw_dataset:
            anon_dataset.append(p.copy())
        for i in range(len(quasi_atts)):
            for person in anon_dataset:
                while DGHs[quasi_atts[i]][person[quasi_atts[i]]].level > (levels[i] - t[i]):
                    person[quasi_atts[i]] = DGHs[quasi_atts[i]][person[quasi_atts[i]]].parent.name
        k_anon = check_k_anon(anon_dataset, k, sensitive_atts)
        if k_anon:
            cost = cost_LM_files(raw_dataset, anon_dataset, DGH_folder)
            results.append((cost, anon_dataset))
            result_lattice = lattice
    
    assert len(results) != 0, "k-anonymity cannot be satisfied"
    results = sorted(results)
    anonymized_dataset = results[0][1]

    write_dataset(anonymized_dataset, output_file)



# Command line argument handling and calling of respective anonymizer:
if len(sys.argv) < 6:
    print(f"Usage: python3 {sys.argv[0]} algorithm DGH-folder raw-dataset.csv anonymized.csv k seed(for random only)")
    print(f"\tWhere algorithm is one of [clustering, random, bottomup]")
    sys.exit(1)

algorithm = sys.argv[1]
if algorithm not in ['clustering', 'random', 'bottomup']:
    print("Invalid algorithm.")
    sys.exit(2)

start_time = datetime.datetime.now() ##
print(start_time) ##

dgh_path = sys.argv[2]
raw_file = sys.argv[3]
anonymized_file = sys.argv[4]
k = int(sys.argv[5])

function = eval(f"{algorithm}_anonymizer");
if function == random_anonymizer:
    if len(sys.argv) < 7:
        print(f"Usage: python3 {sys.argv[0]} algorithm DGH-folder raw-dataset.csv anonymized.csv k seed(for random only)")
        print(f"\tWhere algorithm is one of [clustering, random, bottomup]")
        sys.exit(1)
        
    seed = int(sys.argv[6])
    function(raw_file, dgh_path, k, anonymized_file, seed)
else:    
    function(raw_file, dgh_path, k, anonymized_file)

cost_md = cost_MD(raw_file, anonymized_file, dgh_path)
cost_lm = cost_LM(raw_file, anonymized_file, dgh_path)
print (f"Results of {k}-anonimity:\n\tCost_MD: {cost_md}\n\tCost_LM: {cost_lm}\n")

end_time = datetime.datetime.now() ##
print(end_time) ##
print(end_time - start_time)  ##

# Sample usage:
# python3 main.py clustering DGHs/ adult-hw1.csv result.csv 300 5