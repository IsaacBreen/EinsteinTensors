# from lark import Lark, Transformer, v_args
from re import I
import lark
from lark import Lark
from lark.reconstruct import Reconstructor
# import jax
# from jax import numpy as jnp
# from jax import random
# from jax import jit, vmap, grad, jacobian
import pyperclip
import string
import math
from copy import deepcopy
import functools
from queue import Queue, LifoQueue

grammar = r"""
start: body

body: statement+

statement: (return | equation | function) ("\n" | ";")

return: "return" expression ("," expression)*
equation: value "=" expression
function: "function" IDENTIFIER parameter_tuple body "end"
parameter_tuple: "(" (IDENTIFIER ("," IDENTIFIER)*)? ")"

expression: sum
sum: product (sum_term | sub_term)*
sum_term: "+" product
sub_term: "-" product
product: factor (prod_term | div_term)*
prod_term: "*" factor
div_term: "/" factor
factor: power | value
power: value ("^" value)
value   : PYATTR "(" expression ")" -> function_call
        | "(" expression ")"            -> paren_expr
        | tensor
        | identity_tensor
        | NUMBER
tensor: index "[" indices "]"
identity_tensor: "1" "[" indices "]"
indices: index ("," index)*
index: INDEX_IDENTIFIER ("_" NUMBER)?
PYATTR: IDENTIFIER ("." IDENTIFIER)*
INDEX_IDENTIFIER: LETTER (LETTER|NUMBER)*
IDENTIFIER: LETTER (LETTER|NUMBER|"_")*
LETTER: "a".."z" | "A".."Z"
NUMBER: INTEGER | FLOAT
INTEGER: "0".."9"+
FLOAT: "0".."9"+ "." "0".."9"+

WHITESPACE: (" " | "\n")+
%ignore WHITESPACE
"""

parser = Lark(grammar, maybe_placeholders=False)

s = r"""
function Linear(x)
    y[o] = x[i] * ( x[j] * (x[k] * (x[l] * (x[m] * x[n]))))
    return y[o]
end
"""

tree = parser.parse(s)

def combine_pytrees(tree1, tree2, max_level=-1):
    if isinstance(tree1, dict) and isinstance(tree2, dict):
        keys1 = set(tree1.keys())
        keys2 = set(tree2.keys())
        commonkeys = keys1 & keys2
        common = {key: combine_pytrees(tree1[key], tree2[key], max_level=max_level-1) for key in commonkeys}
        unique1 = {key: tree1[key] for key in keys1 - commonkeys}
        unique2 = {key: tree2[key] for key in keys2 - commonkeys}
        assert isinstance({**common, **unique1, **unique2}, dict)
        return {**common, **unique1, **unique2}
    elif isinstance(tree1, list) and isinstance(tree2, list):
        return tree1 + tree2
    elif isinstance(tree1, tuple) and isinstance(tree2, tuple):
        return tree1 + tree2
    elif isinstance(tree1, set) and isinstance(tree2, set):
        return tree1 | tree2
    else:
        return tree1, tree2

class Visitor:
    def __init__(self, 
                before_callbacks=None,
                after_callbacks=None,
                direction="down",
                reversed=False, 
                traversal_type="breadth-first",
                recursive=None,
                expand_tree_into_callbacks=False,
                trickle_up_accumulator=lambda **trickled_datas: trickled_datas,
                get_children=lambda tree, **_: tree.children if hasattr(tree, "children") else []):
        if recursive is None:
            # If recursive is not specified, use recursion in depth-first traversa and no recursion in breadth-first traversal
            recursive = traversal_type=="depth-first"
        elif recursive:
            assert traversal_type=="depth-first"
        else:
            assert direction=="down"
        assert direction in ["down", "up"], "direction must be 'down' or 'up'"
        assert traversal_type in ["breadth-first", "depth-first"], "traversal_type must be 'breadth-first' or 'depth-first'"
        assert not (direction=="up" and traversal_type=="breadth-first"), "Cannot perform breadth-first traversal when traversing upwards"
        assert traversal_type=="depth-first" or not recursive, "Cannot perform recursive traversal when traversing in breadth-first order"

        if before_callbacks is None:
            before_callbacks = []
        elif callable(before_callbacks):
            # 'before_callbacks' is a single callback function, so wrap it in a list
            before_callbacks = [before_callbacks]
        self.before_callbacks = before_callbacks

        if after_callbacks is None:
            after_callbacks = []
        elif callable(after_callbacks):
            # 'after_callbacks' is a single callback function, so wrap it in a list
            after_callbacks = [after_callbacks]

        self.after_callbacks = after_callbacks
        self.direction = direction
        self.reversed = reversed
        self.traversal_type = traversal_type
        self.recursive = recursive
        self.expand_tree_into_callbacks = expand_tree_into_callbacks
        self.trickle_up_accumulator = trickle_up_accumulator
        self.get_children = get_children

    def register(self, callback_function, i=-1):
        self.before_callbacks.insert(i, callback_function)

    def visit(self, tree, *args, **kwargs):
        if self.recursive:
            return self.visit_recursive(tree, *args, **kwargs)
        else:
            return self.visit_nonrecursive(tree, *args, **kwargs)

    def visit_recursive(self, tree, trickled_data=None, metadata=None):
        assert self.traversal_type=="depth-first", "Cannot perform recursive traversal when traversing in breadth-first order"
        trickled_data = deepcopy(trickled_data) if trickled_data is not None else {}
        metadata = deepcopy(metadata) if metadata is not None else {}
        children = self.get_children(tree)
        if self.reversed:
            children = reversed(children)
        if self.direction == "down":
            return self.visit_downward_dfs(tree, children=children, trickled_data=trickled_data, metadata=metadata)
        if self.direction=="up":
            return self.visit_upward_dfs(tree, children=children, trickled_data=trickled_data, metadata=metadata)


    def visit_downward_dfs(self, tree, children, trickled_data=None, metadata=None):
        # Downward DFS
        # If direction is down, this node has already received the trickled data from above it
        # and is ready to run its callbacks.
        for callback in self.get_before_callbacks():
            new_tree = callback(tree=tree, trickled_data=trickled_data, metadata=metadata)
            if new_tree is not None:
                tree = new_tree
        child_results = []
        for child in children:
            child_result = self.visit_recursive(child, trickled_data, metadata)
            child_results.append(child_result)
        for callback in self.get_after_callbacks():
            new_tree = callback(tree=tree, trickled_data=trickled_data, metadata=metadata, child_results=child_results)
            if new_tree is not None: 
                tree = new_tree
        return tree

    def visit_upward_dfs(self, tree, children, trickled_data=None, metadata=None):
        # Upward DFS
        # The current node needs to accumulate trickled data from its children before running
        # its callbacks. Create a new trickle data accumulator for the children to populate.
        for callback in self.get_before_callbacks():
            new_tree = callback(tree=tree, trickled_data=trickled_data, metadata=metadata)
            if new_tree is not None:
                tree = new_tree
        trickled_data_accumulator = {}
        child_results = []
        for child in children:
            trickled_data_from_child, child_result = self.visit_recursive(child, trickled_data=trickled_data, metadata=metadata)
            trickled_data_accumulator = combine_pytrees(trickled_data_accumulator, trickled_data_from_child)
            child_results.append(child_result)
        trickled_data = self.trickle_up_accumulator(**trickled_data)
        for callback in self.get_after_callbacks():
            new_tree = callback(tree=tree, trickled_data=trickled_data, metadata=metadata, child_results=child_results)
            if new_tree is not None:
                tree = new_tree
        return trickled_data, tree

    def visit_nonrecursive(self, tree, trickled_data=None):
        assert self.direction=="down", "Cannot perform non-recursive traversal when traversing upwards"
        if trickled_data is None: trickled_data = {}
        metadata = {}
        if self.traversal_type=="breadth-first":
            # Use a FILO queue to implement breadth-first traversal
            queue = LifoQueue()
        else:
            # Use a FIFO queue to implement depth-first traversal
            queue = Queue()
        queue.put((tree, trickled_data, metadata, False, None))
        while not queue.empty():
            tree, trickled_data, metadata, already_visited, child_results = queue.get()
            if not already_visited:
                # Run before-callbacks on this node
                for callback in self.get_before_callbacks():
                    new_tree = callback(tree=tree, trickled_data=trickled_data, metadata=metadata)
                    if new_tree is not None:
                        tree = new_tree
                # Queue children
                children = self.get_children(tree)
                if self.reversed:
                    children = reversed(children)
                # Downward BFS and DFS are the same (except that BFD uses a FIFO queue instead of a FILO queue)
                child_results = []
                for child in children:
                    queue.put((child, deepcopy(trickled_data), deepcopy(metadata), False, child_results))
                # Mark this node as visited and reenqueue it
                queue.put((tree, trickled_data, metadata, True, child_results))
            else:
                # Node has already been visited, so run after-callbacks
                for callback in self.get_after_callbacks():
                    new_tree = callback(tree=tree, trickled_data=trickled_data, metadata=metadata, child_results=child_results)
                    if new_tree is not None:
                        tree = new_tree
        return tree

    def get_before_callbacks(self):
        return self.before_callbacks

    def get_after_callbacks(self):
        return self.after_callbacks

def dictionate_function(func):
    def wrapper(tree, *args, **kwargs):
        tree_updates = func(*args, **tree, **kwargs)
        if tree_updates is not None:
            tree.update(tree_updates)
        return tree
    return wrapper

def dictionator(tree, *args, **kwargs):
    """
    Puts a node into a dictionary under the key 'tree'.
    """
    return {'tree': tree}

def antidictionator(tree, *args, **kwargs):
    """
    Reverses the effect of dictionator; extracts the node from the dictionary.
    """
    return tree['tree']

class LarkVisitor(Visitor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def visit(self, tree, *args, **kwargs):
        return super().visit(tree, *args, **kwargs)


def attrfilter(function=None, **attrs):
    attrs = {attr_name: set([attr_values] if isinstance(attr_values, str) else attr_values) for attr_name, attr_values in attrs.items()}
    def _decorate(function):
        @functools.wraps(function)
        def wrapper(tree, *args, **kwargs):
            for attr_name, attr_values in attrs.items():
                if not hasattr(tree, attr_name) or getattr(tree, attr_name) not in attr_values:
                    return
            return function(tree, *args, **kwargs)
        return wrapper
    if function:
        return _decorate(function)
    return _decorate

arithmetic_node_names = ["sum", "product", "factor", "power", "value"]

reconstructor = Reconstructor(parser)

@dictionate_function
@attrfilter(data="indices")
def trickledown_tensor_indices(tree, trickled_data, **kwargs):
    for child in tree.children:
        trickled_data["outer_indices"].add(child)

@dictionate_function
@attrfilter(data="product")
def trickledown_product_indices(tree, trickled_data, **kwargs):
    # print("In trickledown_product_indices")
    print(reconstructor.reconstruct(tree))
    print([reconstructor.reconstruct(index) for index in trickled_data["outer_indices"]])
    print()


trickledown_index_visitor = LarkVisitor(
    direction="up", 
    traversal_type="depth-first", 
    after_callbacks=[dictionator, trickledown_tensor_indices, trickledown_product_indices, antidictionator],
)

initial_trickled_data = {"outer_indices": set()}
tree = trickledown_index_visitor.visit(tree, initial_trickled_data)
print(tree)




