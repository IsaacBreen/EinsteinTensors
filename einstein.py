# from lark import Lark, Transformer, v_args
from re import I
from lark import Lark
import jax
from jax import numpy as jnp
from jax import random
from jax import jit, vmap, grad, jacobian
import pyperclip
import string
import math

l = Lark.open("einstein.lark")# l = Lark(r"""
# start: body

# body: statement+

# statement: (return | equation | function) ("\n" | ";")

# return: "return" expression ("," expression)*
# equation: value "=" expression
# function: "function" IDENTIFIER parameter_tuple body "end"
# parameter_tuple: "(" (IDENTIFIER ("," IDENTIFIER)*)? ")"

# expression: sum
# sum: product (sum_term | sub_term)*
# sum_term: "+" product
# sub_term: "-" product
# product: factor (prod_term | div_term)*
# prod_term: "*" factor
# div_term: "/" factor
# factor: power | value
# power: value ("^" value)
# value   : pyattr "(" expression ")" -> function_call
#         | "(" expression ")"            -> paren_expr
#         | tensor
#         | NUMBER
# tensor: index "[" index ("," index)* "]"
# index: INDEX_IDENTIFIER ("_" NUMBER)?
# pyattr: IDENTIFIER ("." IDENTIFIER)*
# INDEX_IDENTIFIER: LETTER (LETTER|NUMBER)*
# IDENTIFIER: LETTER (LETTER|NUMBER|"_")*
# LETTER: "a".."z" | "A".."Z"
# NUMBER: "0".."9"+

# WHITESPACE: (" " | "\n")+
# %ignore WHITESPACE

class Expression:
    def tensors(self):
        tensors = []
        for child in self.children():
            tensors.extend(child.tensors())
        return tensors


class Sum(Expression):
    def __init__(self, *terms):
        self.terms = terms
    def __repr__(self):
        return f"Sum({', '.join(repr(t) for t in self.terms)})"
    def pystr(self):
        return "( " + " + ".join(t.pystr() for t in self.terms) + " )"
    def children(self):
        return self.terms


class Product(Expression):
    def __init__(self, *factors):
        self.factors = factors
    def __repr__(self):
        return f"Product({', '.join(repr(f) for f in self.factors)})"
    def pystr(self):
        return " * ".join(f.pystr() for f in self.factors)
    def children(self):
        return self.factors


class Power(Expression):
    def __init__(self, base, exponent):
        self.base = base
        self.exponent = exponent
    def __repr__(self):
        return f"Power({repr(self.base)}, {repr(self.exponent)})"
    def pystr(self):
        return f"( {self.base.pystr()} )**( {self.exponent.pystr()} )"
    def children(self):
        return [self.base, self.exponent]


class Tensor(Expression):
    def __init__(self, *indices):
        self.indices = indices
    def __repr__(self):
        return f"Tensor({', '.join(repr(i) for i in self.indices)})"
    def canonical_pyname(self):
        return "_".join(i.basename() for i in self.indices)
    def pystr(self):
        return self.canonical_pyname()
    def children(self):
        return [self]
    def tensors(self):
        return [self]
    def leading_index(self):
        return self.indices[0]
    def nonleading_indices(self):
        return self.indices[1:]
    def __eq__(self, other):
        return isinstance(other, Tensor) and self.indices == other.indices
    def __hash__(self):
        return hash(tuple(self.indices))

class IdentityTensor(Expression):
    def __init__(self, *indices):
        self.indices = indices
    def __repr__(self):
        return f"1({', '.join(repr(i) for i in self.indices)})"
    def pystr(self):
        return self.canonical_pyname()
    def canonical_pyname(self):
        return "1_" + "_".join(i.basename() for i in self.indices)
    def children(self):
        return []
    def tensors(self):
        return []

class Index:
    def __init__(self, name, idx=None):
        self.name = name
        self.idx = idx
    def __repr__(self):
        return f"Index({self.name}, {self.idx})"
    def basename(self):
        return self.name
    def pystr(self):
        return self.name if self.idx is None else f"{self.name}{self.idx}"
    def tensors(self):
        return [self]
    def __eq__(self, other):
        return isinstance(other, Index) and self.name == other.name and self.idx == other.idx
    def __hash__(self):
        return hash((self.name, self.idx))


class Number:
    def __init__(self, value):
        self.value = value
    def pystr(self):
        return str(self.value)
    def tensors(self):
        return []


class Identifier:
    def __init__(self, name):
        self.name = name
    def __repr__(self):
        return f"Identifier({self.name})"
    def pystr(self):
        return self.name
    def tensors(self):
        return []


class FunctionCall(Expression):
    def __init__(self, name, *args):
        self.name = name
        self.args = args
    def __repr__(self):
        return f"FunctionCall({self.name}, {', '.join(repr(a) for a in self.args)})"
    def pystr(self):
        return f"{self.name}( {', '.join(a.pystr() for a in self.args)} )"
    def children(self):
        return self.args


class Return(Expression):
    def __init__(self, *values):
        self.values = values
    def __repr__(self):
        return f"Return({', '.join(repr(v) for v in self.values)})"
    def pystr(self):
        return f"return {', '.join(v.pystr() for v in self.values)}"
    def children(self):
        return self.values


class Equation:
    def __init__(self, lhs, rhs):
        self.lhs = lhs
        self.rhs = rhs
    def __repr__(self):
        return f"Equation({repr(self.lhs)}, {repr(self.rhs)})"

    def make_apply(self):
        tensors = self.rhs.tensors()
        s = ""
        return s

    def tensors(self):
        tensors = []
        tensors.extend(self.lhs.tensors())
        tensors.extend(self.rhs.tensors())
        return tensors

    
class Function:
    def __init__(self, name, arguments, body):
        self.name = name
        self.arguments = arguments
        self.body = body
    def __repr__(self):
        return f"Function({self.name}, {', '.join(repr(a) for a in self.arguments)}, {repr(self.body)})"
    def pystr(self):
        return f"def {self.name}({', '.join(a.pystr() for a in self.arguments)}):\n{self.body.pystr()}"
    def tensors(self):
        return []

class Equations:
    def __init__(self, *equations):
        self.equations = [equations for equations in equations if isinstance(equations, Equation)]
        self.returns = [equations for equations in equations if isinstance(equations, Return)]
        self.functions = [equations for equations in equations if isinstance(equations, Function)]
    def __repr__(self):
        return f"Equations({', '.join(repr(e) for e in self.equations)})"
    def make_apply(self):
        s = ""
        for e in self.equations:
            s += e.make_apply() + "\n"
        return s

    def tensors(self):
        tensors = []
        for e in self.equations:
            tensors.extend(e.lhs.tensors())
            tensors.extend(e.rhs.tensors())
        return tensors


def run_tree(tree):
    if hasattr(tree, "data"):
        if tree.data == 'start':
            return run_tree(tree.children[0])
        elif tree.data == 'body':
            return Equations(*[run_tree(e) for e in tree.children])
        elif tree.data == 'parameter_tuple':
            return [run_tree(e) for e in tree.children]
        elif tree.data == 'statement':
            return run_tree(tree.children[0])
        elif tree.data == 'return':
            return Return(*[run_tree(e) for e in tree.children])
        elif tree.data == 'function':
            return Function(run_tree(tree.children[0]), run_tree(tree.children[1]), run_tree(tree.children[2]))
        elif tree.data == 'equation':
            lhs = run_tree(tree.children[0])
            rhs = run_tree(tree.children[1])
            return Equation(lhs, rhs)
        elif tree.data == 'expression':
            return run_tree(tree.children[0])
        elif tree.data == 'sum':
            if len(tree.children) == 0:
                return Number(0)
            elif len(tree.children) == 1:
                return run_tree(tree.children[0])
            else:
                return Sum(*[run_tree(t) for t in tree.children])
        elif tree.data == 'sum_term':
            return run_tree(tree.children[0])
        elif tree.data == 'sub_term':
            return Product(run_tree(tree.children[0]), Number(-1))
        elif tree.data == 'product':
            if len(tree.children) == 0:
                return Number(1)
            elif len(tree.children) == 1:
                return run_tree(tree.children[0])
            else:
                return Product(*[run_tree(t) for t in tree.children])
        elif tree.data == 'prod_term':
            return run_tree(tree.children[0])
        elif tree.data == 'div_term':
            return Power(run_tree(tree.children[0]), Number(-1))
        elif tree.data == 'factor':
            return run_tree(tree.children[0])
        elif tree.data == 'power':
            return Power(run_tree(tree.children[0]), run_tree(tree.children[1]))
        elif tree.data == 'function_call':
            name = run_tree(tree.children[0])
            args = [run_tree(t) for t in tree.children[1:]]
            return FunctionCall(name, *args)
        elif tree.data == 'paren_expr':
            return run_tree(tree.children[0])
        elif tree.data == 'tensor':
            return Tensor(*[run_tree(t) for t in tree.children])
        elif tree.data == 'identity_tensor':
            return IdentityTensor(*[run_tree(t) for t in tree.children])
        elif tree.data == 'value':
            return run_tree(tree.children[0])
        elif tree.data == 'tensor':
            return Tensor(*[run_tree(t) for t in tree.children])
        elif tree.data == 'index':
            name = run_tree(tree.children[0])
            if len(tree.children) > 1:
                idx = run_tree(tree.children[1]).value
            else:
                idx = None
            return Index(name, idx)
    elif tree.type == 'PYATTR':
        return tree.value
    elif tree.type == 'IDENTIFIER':
        return tree.value
    elif tree.type == 'INDEX_IDENTIFIER':
        return tree.value
    elif tree.type == 'LETTER':
        return tree.value
    elif tree.type == 'NUMBER':
        return Number(tree.value)
    raise ValueError(f"Unknown tree: {tree}")
    


# eqs = run_tree(tree)
# print(eqs)
# print(eqs.make_apply())
# print(eqs.tensors())

def get_transexpand_and_collapse_instructions(eq):
    ordered_indices = []
    lhs_tensor = eq.lhs.tensors()[0]
    for index in lhs_tensor.nonleading_indices():
        ordered_indices.append(index)
    for tensor in eq.rhs.tensors():
        for index in tensor.nonleading_indices():
            if index not in ordered_indices:
                ordered_indices.append(index)

    lhs_indices = list(lhs_tensor.nonleading_indices())
    dimensions_for_final_collapse_to_lhs = []
    for i, index in enumerate(ordered_indices):
        if index not in lhs_indices:
            dimensions_for_final_collapse_to_lhs.append(i)

    dimensions_for_transpose_and_expand = {}
    for tensor in eq.rhs.tensors():
        dims = []
        current_indices = tensor.nonleading_indices()
        for index in ordered_indices:
            if index in current_indices:
                dims.append(current_indices.index(index))
            else:
                dims.append(None)
        dimensions_for_transpose_and_expand[tensor] = dims
    return dimensions_for_transpose_and_expand, dimensions_for_final_collapse_to_lhs


def transexpand(tensor, dims):
    """
    Combines transpose and expand_dims.

    >>> x = np.array([[1, 2, 3],[4, 5, 6]])
    >>> x.shape
    (2, 3)
    >>> transexpand(x, [1, None, 0]).shape
    (3, 1, 2)
    """
    numbered_dims = [dim for dim in dims if dim is not None]
    if set(numbered_dims) - {None} != set(range(tensor.ndim)):
        raise TypeError(f"transpose permutation isn't a permutation of operand dimensions, got permutation {dims} for operand with shape {tensor.shape}.")
    return tensor.transpose(numbered_dims).__getitem__(tuple(None if dim is None else slice(None) for dim in dims))

def codegen_transexpand(tensor_name, dims):
    """
    >>> codegen_transexpand("x", [1, None, 0])
    'x = x.transpose(1, 0)[:, None, :]'
    """
    s = ""
    s += tensor_name
    numbered_dims = tuple(dim for dim in dims if dim is not None)
    if numbered_dims != tuple(range(len(numbered_dims))):
        s += f".transpose({numbered_dims})"
    at_least_one_expansion = False
    for dim in dims:
        if dim is None:
            at_least_one_expansion=True
    if at_least_one_expansion:
        index_str = ', '.join("None" if dim is None else ":" for dim in dims)
        s += f"[{index_str}]" if index_str else ""
    return s

def codegen_collapse(tensor_name, dims, keepdims=False):
    """
    >>> codegen_collapse("x", [0, -1])
    'x = np.sum(x, axis=(0, -1))'
    """
    return f"{tensor_name}.sum(axis={dims}{', keepdims=True' if keepdims else ''})"

# def construct_equation(eq, parameter_tensors=None):
#     """
#     >>> eq = Equation(Tensor(Index('y', None), Index('b', None), Index('i', None)), Product(Tensor(Index('w', None), Index('i', 1), Index('i', None)), Tensor(Index('x', None), Index('b', None), Index('i', 1))))
#     >>> construct_equation(eq)
#     'y_b_i = (w_i_i1.transpose(1, 0)[None, :, :] * x_b_i1.transpose(0, 1)[:, None, :]).sum(axis=(1,))'
#     """
#     transpand_dims, collapse_dims = get_transexpand_and_collapse_instructions(eq)
    
#     symbols = {
#         Sum: {'symbol': '+', 'is_pycall': False, 'precedence': 4},
#         Product: {'symbol': '*', 'is_pycall': False, 'precedence': 3},
#         Power: {'symbol': '**', 'is_pycall': False, 'precedence': 2},
#         FunctionCall: {'symbol': '', 'is_pycall': True, 'precedence': 1},
#     }
#     def helper(e, prev_precedence):
#         if isinstance(e, Number):
#             return str(e.value)
#         elif isinstance(e, Tensor):
#             tensor_string = e.pystr()
#             if e in parameter_tensors:
#                 tensor_string = f"params['{tensor_string}']"
#             return codegen_transexpand(tensor_string, transpand_dims[e])
#         symbol_data = symbols[type(e)]
#         if symbol_data['is_pycall']:
#             return f"{e.name}({', '.join(helper(child, symbol_data['precedence']) for child in e.children())})"
#         else:
#             if prev_precedence < symbol_data['precedence']:
#                 return f"({helper(e, symbol_data['precedence'])})"
#             else:
#                 if isinstance(e, Sum):
#                     ss = []
#                     for child in e.children():
#                         s = f"({helper(child, symbol_data['precedence'])})"
#                         s = codegen_collapse(s, collapse_dims, keepdims=True)
#                         ss.append(s)
#                     return symbol_data['symbol'].join(ss)
#                 return symbol_data['symbol'].join(helper(child, symbol_data['precedence']) for child in e.children())
#     return f"{eq.lhs.pystr()} = {codegen_collapse('(' + helper(eq.rhs, 0) + ')', collapse_dims)}"

def construct_equation(eq, parameter_tensors=None, sizes_required=None):
    """
    >>> eq = Equation(Tensor(Index('y', None), Index('i', 1)), Sum(Product(Tensor(Index('w', None), Index('i', 1), Index('i', 2)), Tensor(Index('x', None), Index('i', 2))), Tensor(Index('b', None), Index('i', 1))))
    >>> construct_equation(eq)
    'y_1 = np.einsum("ij, j->i", w_i_i, x_i) + b_i'
    """

    symbols = {
        Sum: {'symbol': ' + ', 'is_pycall': False, 'precedence': 4},
        Product: {'symbol': '*', 'is_pycall': False, 'precedence': 10},
        Power: {'symbol': '**', 'is_pycall': False, 'precedence': 2},
        FunctionCall: {'symbol': '', 'is_pycall': True, 'precedence': 10},
    }

    output_tensor = eq.lhs

    def helper(eq, prev_precedence, outer_indices):
        if isinstance(eq, Number):
            return str(eq.value), [], {'is_tensor': False, 'collapsable': True}
        elif isinstance(eq, Tensor):
            tensor_string = eq.pystr()
            if eq in parameter_tensors:
                tensor_string = f"params['{tensor_string}']"
            return tensor_string, eq.nonleading_indices(), {'is_tensor': True, 'collapsable': True}
        elif isinstance(eq, IdentityTensor):
            immediately_collapsable = [index.basename() for index in set(eq.indices) - set(outer_indices)]
            immediately_collapsable_str = "*".join(immediately_collapsable) if immediately_collapsable else None
            if sizes_required is not None:
                sizes_required.update(immediately_collapsable)
            collapse_later = [index for index in set(eq.indices) if index in outer_indices]
            return immediately_collapsable_str, collapse_later, {'is_tensor': False, 'collapsable': True}
        elif symbols[type(eq)]['precedence'] > prev_precedence:
            s, output_index, hret = helper(eq, symbols[type(eq)]['precedence'], outer_indices)
            return f"({s})", output_index, hret
        elif isinstance(eq, Product):
            def get_all_indices(expression):
                indices = []
                for child in expression.children():
                    if isinstance(child, Tensor):
                        indices.extend(child.nonleading_indices())
                    elif isinstance(child, (Sum, Product, Power, FunctionCall)):
                        indices.extend(get_all_indices(child))
                return indices
            child_cumulative_indices = [get_all_indices(child) for child in eq.children()]
            all_child_indices = set()
            for child_cumulative_indices_ in child_cumulative_indices:
                all_child_indices.update(child_cumulative_indices_)
            outer_and_child_indices = all_child_indices.union(outer_indices)
            child_strings = []
            child_output_indices = []
            output_index = []
            is_tensor = False
            collapsable = False
            hdicts = []
            for child in eq.children():
                outer_and_other_child_indices = set(outer_indices)
                for child2 in eq.children():
                    if child2 is not child:
                        outer_and_other_child_indices.update(get_all_indices(child2))
                child_string, child_output_index, hdict = helper(child, symbols[type(eq)]['precedence'], outer_and_other_child_indices)
                if child_string:
                    child_strings.append(child_string)
                    child_output_indices.append(child_output_index)
                    hdicts.append(hdict)
                if hdict['collapsable']:
                    for index in child_output_index:
                        if index in outer_indices and index not in output_index:
                            output_index.append(index)
                is_tensor = is_tensor or hdict['is_tensor']
                collapsable = collapsable or hdict['collapsable']
            output_index = [index for index in output_tensor.nonleading_indices() if index in output_index] + [index for index in output_index if index not in output_tensor.nonleading_indices()]
            all_child_output_indices = set()
            for child_output_indices_ in child_output_indices:
                all_child_output_indices.update(child_output_indices_)
            input_and_output_indices = all_child_output_indices.union(output_index)
            # Assign each index a letter
            index_letters = {}
            n_dupes = 0
            for index in input_and_output_indices:
                if index.basename()[0] in index_letters.values():
                    index_letters[index] = string.ascii_lowercase[n_dupes]
                    n_dupes += 1
                else:
                    index_letters[index] = index.basename()[0]
            einstring_left = []
            tensor_child_strings = []
            nontensor_child_strings = []
            for child_string, indices, hdict in zip(child_strings, child_output_indices, hdicts):
                if indices is not None and hdict['is_tensor']:
                    einindex = ""
                    for index in indices:
                        einindex += index_letters[index]
                    einstring_left.append(einindex)
                if hdict['is_tensor']:
                    tensor_child_strings.append(child_string)
                else:
                    nontensor_child_strings.append(child_string)
            einstring_right = "".join(index_letters[index] for index in output_index)
            einstring = ",".join(s for s in einstring_left) + "->" + einstring_right
            return f"{''.join('(' + s + ')*' for s in nontensor_child_strings)}jnp.einsum('{einstring}', {', '.join(tensor_child_strings)})", output_index, {'is_tensor': is_tensor, 'collapsable': collapsable}
        elif isinstance(eq, (Sum, Power)):
            child_strings = []
            child_output_indices = []
            output_index = []
            is_tensor = False
            collapsable = False
            for child in eq.children():
                child_string, child_output_index, hdict = helper(child, symbols[type(eq)]['precedence'], outer_indices)
                if child_string:
                    child_strings.append(child_string)
                    child_output_indices.append(child_output_index)
                # if hdict['collapsable']:
                for index in child_output_index:
                    # if index in outer_indices and index not in output_index:
                    if index not in output_index:
                        output_index.append(index)
                is_tensor = is_tensor or hdict['is_tensor']
                collapsable = collapsable or hdict['collapsable']
            child_strings_transexpanded = []
            for child_string, child_output_indices in zip(child_strings, child_output_indices):
                if child_output_indices:
                    child_output_indices_pos = {index: i for i, index in enumerate(child_output_indices)}
                    transexpand_dims = [child_output_indices_pos[index] if index in child_output_indices_pos else None for index in output_index]
                    child_strings_transexpanded.append(codegen_transexpand(child_string, transexpand_dims))
                else:
                    child_strings_transexpanded.append(child_string)
            return symbols[type(eq)]['symbol'].join(child_strings_transexpanded), output_index, {'is_tensor': is_tensor, 'collapsable': collapsable}
        elif isinstance(eq, FunctionCall):
            child_strings = []
            child_output_indices = []
            is_tensor = False
            collapsable = False
            for child in eq.children():
                child_string, child_output_index, hdict = helper(child, symbols[type(eq)]['precedence'], outer_indices)
                if child_string:
                    child_strings.append(child_string)
                    child_output_indices.append(child_output_index)
                is_tensor = is_tensor or hdict['is_tensor']
                collapsable = collapsable or hdict['collapsable']
            # TODO: this is a dodgy way of assuming the output shape
            output_index = child_output_indices[0]
            return f"{eq.name}({', '.join(child_strings)})", output_index, {'is_tensor': is_tensor, 'collapsable': collapsable}
        else:
            raise ValueError(f"Unknown expression type {type(eq)}")
        raise ValueError(f"This should never happen; we should have handled all cases.")

    outer_indices = [index for index in eq.lhs.nonleading_indices()]
    rhs_string = helper(eq.rhs, math.inf, outer_indices)[0]
    return f"{eq.lhs.pystr()} = {rhs_string}"

        
def construct_equations(eqs, parameter_tensors=None, sizes_required=None):
    return "\n".join(construct_equation(eq, parameter_tensors=parameter_tensors, sizes_required=sizes_required) for eq in eqs.equations)

def split_tensors(eqs, inputs):
    parameter_tensors = []
    input_tensors = []
    seen_tensors = set()
    for eq in eqs.equations:
        for tensor in eq.rhs.tensors():
            if tensor.canonical_pyname() not in seen_tensors:
                if tensor.leading_index().basename() in inputs:
                    input_tensors.append(tensor)
                else:
                    parameter_tensors.append(tensor)
            seen_tensors.add(tensor.canonical_pyname())
        seen_tensors.add((eq.lhs.tensors())[0].pystr())
    return parameter_tensors, input_tensors

def split_indices(parameter_tensors, input_tensors):
    input_indices = set()
    for tensor in input_tensors:
        input_indices.update([index.basename() for index in tensor.nonleading_indices()])
    parameter_indices = set()
    for tensor in parameter_tensors:
        parameter_indices.update([index.basename() for index in tensor.nonleading_indices()])
    parameter_indices -= input_indices
    return parameter_indices, input_indices

def construct_jax_module_from_equations(name, eqs, inputs, jit):
    parameter_tensors, input_tensors = split_tensors(eqs, inputs)
    parameter_indices, input_indices = split_indices(parameter_tensors, input_tensors)
    parameter_arg_list = ', '.join(index for index in parameter_indices)
    parameter_kwarg_list = ', '.join(f"{index}={index}" for index in parameter_indices)
    s = f"class {name}:\n"
    s +=f"    def __init__(self, {parameter_arg_list}):\n"
    dim_sizes = ', '.join(f"'{index}': {index}" for index in parameter_indices)
    s += f"        self.old_init = self.init\n"
    s += f"        self.init = lambda *args, **kwargs: self.old_init(*args, {parameter_kwarg_list}, **kwargs)\n"
    s += "\n"
    s += "    " + construct_jax_init_from_equations(eqs, inputs).replace("\n", "\n    ")
    s += "\n"
    s += "    @staticmethod\n"
    s += "    " + construct_jax_apply_from_equations(eqs=eqs, inputs=inputs, jit=jit).replace("\n", "\n    ")
    return s

def construct_jax_init_from_equations(eqs, inputs):
    parameter_tensors, input_tensors = split_tensors(eqs, inputs)
    parameter_indices, input_indices = split_indices(parameter_tensors, input_tensors)
    s = ""
    s +="@staticmethod\n"
    s += f"def init(key, {''.join(index + ', ' for index in list(input_indices) + list(parameter_indices))}**kwargs):\n"
    s += f"    keys = jax.random.split(key, {len(parameter_tensors)})\n"
    longest_name = max(len(tensor.canonical_pyname()) for tensor in parameter_tensors)
    tensor_defs = []
    for i, tensor in enumerate(parameter_tensors):
        shape_string = "[" + ', '.join(index.basename() for index in tensor.nonleading_indices()) + "]"
        tensor_defs.append({
            "name": tensor.canonical_pyname(),
            "key_name": f'"{tensor.canonical_pyname()}": ',
            "value": f"jax.random.normal(keys[{i}], shape={shape_string})"})
    max_name_len = max(len(tensor_def['key_name']) for tensor_def in tensor_defs)
    param_dict_inner_str = "        " + ",\n        ".join(tensor_def['key_name'].ljust(max_name_len) + tensor_def['value'] for tensor_def in tensor_defs)
    s += f"    return {{\n{param_dict_inner_str}\n    }}\n"
    return s

def construct_jax_apply_from_equations(eqs, inputs, jit=True):
    parameter_tensors, input_tensors = split_tensors(eqs, inputs)
    parameter_indices, input_indices = split_indices(parameter_tensors, input_tensors)
    input_tensor_names = [tensor.pystr() for tensor in input_tensors]
    parameter_tensor_names = [tensor.pystr() for tensor in parameter_tensors]
    s = ""
    # s +=f"@lambda apply: vmap(apply, in_axes=({', '.join(['0' for _ in input_tensors] + ['None' for _ in parameter_tensors])}))\n"
    s += "@jit\n" if jit else ""
    s +=f"@lambda apply: vmap(apply, in_axes=({', '.join(['0' for _ in input_tensors])}, None))\n"
    s += f"def apply({''.join([name + ', ' for name in input_tensor_names])}params):\n"
    # Indent
    sizes_required = set()
    eq_string = construct_equations(eqs, parameter_tensors, sizes_required=sizes_required).replace("\n", "\n    ")
    if len(sizes_required)>0:
        sizes = []
        for index_basename in sizes_required:
            break_flag = False
            for tensor in input_tensors:
                for i, index in enumerate(tensor.nonleading_indices()):
                    if index.basename() == index_basename:
                        sizes.append(f"{tensor.pystr()}.shape[{i}]")
                        break_flag = True
                        break
                if break_flag:
                    break
            if break_flag:
                continue
            for tensor in parameter_tensors:
                for i, index in enumerate(tensor.nonleading_indices()):
                    if index.basename() == index_basename:
                        sizes.append(f"params['{tensor.canonical_pyname()}'].shape[{i}]")
                        break_flag = True
                        break
                if break_flag:
                    break
            if not break_flag:
                raise ValueError(f"Could not find index {index_basename} in {input_tensors + parameter_tensors}")
        s += "    " + ", ".join(sizes_required) + " = " + ", ".join(sizes) + "\n"
    s += "    " + eq_string + "\n"
    s += "    " + eqs.returns[0].pystr()
    return s

def construct_jax_modules(body, main_name="main", main_inputs=None, jit=True):
    s = []
    functions = body.functions
    leftover_expressions = body
    for function in functions:
        s.append(construct_jax_module_from_equations(function.name, function.body, function.arguments, jit=jit))
    s = '\n\n\n'.join(s)
    s += construct_jax_module_from_equations(main_name, leftover_expressions, main_inputs if main_inputs is not None else [], jit=jit) if leftover_expressions.returns else ""
    return s

# print(construct_jax_module_from_equations("TestModule", eqs, "x", False))

def jax_codegen(s, module_name="main", main_inputs=None, jit=True):
    tree = l.parse(s)
    eqs = run_tree(tree)
    c = construct_jax_modules(eqs, main_inputs, jit=jit)
    return c

def build_jax_module(s, jit=True):
    tree = l.parse(s)
    eqs = run_tree(tree)
    c = construct_jax_modules(eqs, jit=jit)
    exec(c)
    module_names = [function.name for function in eqs.functions]
    return eval(f"{', '.join(module_names)}")


# s = """
# y i = w i j * x i + b i;
# return y i;
# """

# s = """
# y.i = w.i.j * x.i + b.i;
# return y i;
# """

# s = """
# y,i = w,i,j * x,i + b,i;
# return y,i;
# """

# s = """
# y[i] = w[i,j] * x[i] + b[i]
# return y[i]
# """


# s = """
# function Linear(x)
#     y[j] = w[i,j] * x[i] + b[j]
#     return y[j]
# end
# """

s = """
function Linear(x)
    y[o] = x[i] * w[i,j] * w[j,o]
    return y[o]
end
"""

# s = """
# function Linear(x)
#     z[o] = w[o,i] * x[i] + b[o] + 1[i] + x[i]*1[o]
#     return z[o]
# end
# """

# s = """
# function Linear(x)
#     z[o] = w[o,i] * x[i] + b[o] + 1[o] + 1[i] + 1[o,i] + x[i]*1[o,i] + x[i]*1[o]
#     return z[o]
# end
# """

# s = """
# function MultiheadSelfAttention(x)
#     x[t,i] = x[t,i] * gx[i] * (x[t,i_1]^2 * 1[i_1] / (x[t,i_2] * 1[i_2]))^0.5
#     q[h,t,j] = q[h,j,i] * x[t,i]
#     k[h,t,j] = k[h,j,i] * x[t,i]
#     v[h,t,j] = v[h,j,i] * x[t,i]
#     a[h,t_1,t_2] = jnp.exp(q[h,t_1,j] * k[h,t_2,j])
#     u[t,k] = activation(wu[h,j,k] * a[h,t,t_2] * v[h,t_2,j] + bu[k])
#     z[t,i] = wz[i,k] * u[t,k] + bz[i]
#     z[t,i] = z[t,i] * gz[i] * (z[t,i]^2 * 1[i] / (z[t,i] + 1[i]))
#     return z[t,i]
# end
# """

if __name__ == "__main__":
    print(jax_codegen(s))

# function MultiheadSelfAttention(x)
#     x[t,i] = x[t,i] * gx[i]
#     q[h,t,j] = q[h,j,i] * x[t,i]
#     k[h,t,j] = k[h,j,i] * x[t,i]
#     v[h,t,j] = v[h,j,i] * x[t,i]
#     a[h,t_1,t_2] = jnp.exp(q[h,t_1,j] * k[h,t_2,j])
#     u[t,k] = activation(wu[h,j,k] * a[h,t,t_2] * v[h,t_2,j] + bu[k])
#     z[t,i] = wz[t,k] * u[t,k] + bz[i]
#     return z[t,i]
# end
# """

# print(jax_codegen(s))
# print(construct_jax_modules(run_))




# print_tests()
