import math
import cmath
import numpy as np
from dataclasses import dataclass
from typing import NamedTuple
from dataclasses import dataclass
import networkx as nx

# qclib.state_preparation.util.tree_utils

def create_angles_tree(state_tree):
    """
    :param state_tree: state_tree is an output of state_decomposition function
    :param tree: used in the recursive calls
    :return: tree with angles that will be used to perform the state preparation
    """
    mag = 0.0
    if state_tree.mag != 0.0:
        mag = state_tree.right.mag / state_tree.mag

    arg = state_tree.right.arg - state_tree.arg

    # Avoid out-of-domain value due to numerical error.
    if mag < -1.0:
        angle_y = -math.pi
    elif mag > 1.0:
        angle_y = math.pi
    else:
        angle_y = 2 * math.asin(mag)
        while angle_y < 0:
            angle_y = angle_y+4*math.pi
        #while angle_y > 2*math.pi:
        #    angle_y = angle_y-2*math.pi


    angle_z = 2 * arg
    while angle_z < 0:
        angle_z = angle_z+4*math.pi
    #while angle_z > 2*math.pi:
    #    angle_z = angle_z-4*math.pi

    node = NodeAngleTree(state_tree.index, state_tree.level, angle_y, angle_z, None, None)

    if not is_leaf(state_tree.left):
        node.right = create_angles_tree(state_tree.right)
        node.left = create_angles_tree(state_tree.left)

    return node

def is_leaf(tree):
    """
    :param tree: a tree node
    :return: True if tree is a leaf
    """
    if tree.left is None and tree.right is None:
        return True

    return False

def remove_leafs(tree):
    """remove tree leafs"""
    if tree.left:
        if is_leaf(tree.left):
            tree.left = None
        else:
            remove_leafs(tree.left)

    if tree.right:
        if is_leaf(tree.right):
            tree.right = None
        else:
            remove_leafs(tree.right)

def leftmost(tree):
    """
    :param tree: a tree node
    :return: the leftmost node relative to tree, or None if tree is leaf.
    """
    if tree.left:
        return tree.left

    return tree.right

def node_index(tree):
    """
    :param tree: a tree node
    :return: the total index of the node in the tree.
    """
    return 2**tree.level - 1 + tree.index

def root_node(tree, level):
    """
    :param tree: a tree node
    :param level: level of the subtree (0 for the tree root)
    :return: subtree root at level
    """
    root = tree
    while root.level > level:
        root = root.parent

    return root

def children(nodes):
    """
    Search and list all the nodes childs.
    :param nodes: a list with tree nodes
    :return: a list with nodes childs
    """
    child = []
    for node in nodes:
        if node.left:
            child.append(node.left)
        if node.right:
            child.append(node.right)

    return child

def length(tree):
    """
    Count the total number of the tree nodes.
    :param tree: a tree node
    :return: the total of nodes in the subtree
    """
    if tree:
        n_nodes = length(tree.left)
        n_nodes += length(tree.right)
        n_nodes += 1
        return n_nodes
    return 0

def level_length(tree, level):
    """
    Count the total number of the tree nodes in the level.
    :param tree: a tree node
    :param level: a tree level
    :return: the total of nodes in the subtree level
    """
    if tree:
        if tree.level < level:
            n_nodes_level = level_length(tree.left, level)
            n_nodes_level += level_length(tree.right, level)
            return n_nodes_level

        return 1

    return 0

def height(root):
    """
    Count the number of levels in the tree.
    :param root: subtree root node
    :return: the total of levels in the subtree defined by root
    """
    n_levels = 0
    left = root
    while left:
        n_levels += 1
        left = leftmost(left)

    return n_levels

def left_view(root, stop_level):
    """
    :param root: subtree root node
    :param stop_level: level below root to stop the search
    :return: list of leftmost nodes between root level and stop_level
    """
    branch = []
    left = root
    while left and left.level <= stop_level:
        branch.append(left)
        left = leftmost(left)

    return branch

def subtree_level_index(root, tree):
    """
    :param root: subtree root node
    :param tree: a tree node
    :return: the index of tree node repective to the subtree defined by root
    """
    return tree.index - root.index * 2 ** (tree.level - root.level)

def subtree_level_leftmost(root, level):
    """
    :param root: subtree root node
    :param level: level to search for the leftmost node
    :return: the leftmost tree node repective to the subtree defined by root
    """
    left = root
    while left and left.level < level:
        left = leftmost(left)
    return left

def subtree_level_nodes(tree, level, level_nodes):
    """
    Search and list all the nodes in the indicated level of the tree defined by
    the first value of tree (subtree root).
    :param tree: current tree node, starts with subtree root node
    :param level: level to search for the nodes
    :out param level_nodes: a list with the level tree nodes repective to the
                            subtree defined by root, ordered from left to right
    """
    if tree.level < level:
        if tree.left:
            subtree_level_nodes(tree.left, level, level_nodes)
        if tree.right:
            subtree_level_nodes(tree.right, level, level_nodes)
    else:
        level_nodes.append(tree)

def tree_visual_representation(tree, dot=None):
    """
    :param tree: A binary tree, with str(tree) defined
    """

    if dot is None:
        dot = nx.Digraph()
        dot.node(str(tree))

    if tree.left:
        dot.node(str(tree.left))
        dot.edge(str(tree), str(tree.left))
        dot = tree_visual_representation(tree.left, dot=dot)

    if tree.right:
        dot.node(str(tree.right))
        dot.edge(str(tree), str(tree.right))
        dot = tree_visual_representation(tree.right, dot=dot)

    return dot



#state_decomposition, Amplitude
class Amplitude(NamedTuple):
    """
    Named tuple for amplitudes
    """

    index: int
    amplitude: float

    def __str__(self):
        return f"{self.index}:{self.amplitude:.2f}"

@dataclass
class Node:
    """
    Binary tree node used in state_decomposition function
    """

    index: int
    level: int
    left: "Node"
    right: "Node"
    mag: float
    arg: float
    

    def __str__(self):
        return (
            f"{self.level}_"
            f"{self.index}\n"
            f"{self.mag:.2f}_"
            f"{self.arg:.2f}"
        )
          
def state_decomposition(nqubits, data):
    """
    :param nqubits: number of qubits required to generate a
                    state with the same length as the data vector (2^nqubits)
    :param data: list with exactly 2^nqubits pairs (index, amplitude)
    :return: root of the state tree
    """
    new_nodes = []

    ## leafs
    #for k in data:
    #    new_nodes.append(Node(k.index, nqubits, None, None, abs(k.amplitude), cmath.phase(k.amplitude)))
    for k in data:
        myphase = cmath.phase(k.amplitude)
        while myphase < 0:
            myphase = myphase+2*np.pi
        while myphase > 2*np.pi:
            myphase = myphase-2*np.pi
        new_nodes.append(Node(k.index, nqubits, None, None, abs(k.amplitude), myphase))



    # build state tree
    while nqubits > 0:
        nodes = new_nodes
        new_nodes = []
        nqubits = nqubits - 1
        k = 0
        n_nodes = len(nodes)
        while k < n_nodes:
            mag = math.sqrt(nodes[k].mag ** 2 + nodes[k + 1].mag ** 2)
            arg = (nodes[k].arg + nodes[k + 1].arg) / 2
            new_nodes.append(Node(nodes[k].index // 2, nqubits, nodes[k], nodes[k + 1], mag, arg))
            k = k + 2

    tree_root = new_nodes[0]
    return tree_root

@dataclass
class NodeAngleTree:
    """
    Binary tree node used in function create_angles_tree
    """

    index: int
    level: int
    angle_y: float
    angle_z: float
    left: "NodeAngleTree"
    right: "NodeAngleTree"
    

    def __str__(self):
        space = '\t' * self.level
        txt = f"{space * self.level} y {self.angle_y:.2f} z{self.angle_z:.2f}\n"
        if self.left is not None:
            txt += self.left.__str__()
            txt += self.right.__str__()
        return txt

    def update_index(self, node_to_update, new_node):
        #node_to_update = self.find(old_value)
        if node_to_update:
            node_to_update.index = new_node.index
        print("nodetoup:",node_to_update.index, new_node.index)
        
   