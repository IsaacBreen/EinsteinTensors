# from lark import Lark, Transformer, v_args
from re import I
from lark import Lark
import jax
from jax import numpy as jnp
from jax import random
from jax import jit, vmap, grad, jacobian
import string
import math
from pathlib import Path

l = Lark.open(Path(__file__).parent / "einstein.lark")


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


class CompoundTensor(Expression):
    def __init__(self, *indices):
        self.indices = indices

    def __repr__(self):
        return f"CompoundTensor({', '.join(repr(i) for i in self.indices)})"

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

    def nonleading_indices(self, expand_compounds=True):
        if expand_compounds:
            ret = []
            for index in self.indices[1:]:
                if isinstance(index, CompoundIndex):
                    for in_index in index.in_indices:
                        ret.append(in_index)
                else:
                    ret.append(index)
            return ret
        else:
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


class CompoundIndex:
    def __init__(self, in_indices, out_index):
        self.in_indices = in_indices
        self.out_index = out_index

    def __repr__(self):
        return f"CompoundIndex({self.in_indices}, {self.out_index})"

    def basename(self):
        return self.out_index.basename()

    def pystr(self):
        return f"{self.out_index.pystr()}[{', '.join(i.pystr() for i in self.in_indices)}]"

    def tensors(self):
        return [self.out_index]

    def __eq__(self, other):
        return isinstance(other, CompoundIndex) and self.in_indices == other.in_indices and self.out_index == other.out_index

    def __hash__(self):
        return hash((*self.in_indices, self.out_index))


class Number:
    def __init__(self, value):
        self.value = value

    def pystr(self):
        return str(self.value)

    def tensors(self):
        return []

    def children(self):
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
        self.equations = [
            equations for equations in equations if isinstance(equations, Equation)]
        self.returns = [
            equations for equations in equations if isinstance(equations, Return)]
        self.functions = [
            equations for equations in equations if isinstance(equations, Function)]

        if self.returns:
            equations_map = {
                eq.lhs.canonical_pyname(): eq for eq in self.equations}
            equation_names = [eq.lhs.canonical_pyname()
                              for eq in self.equations]

            def helper(expr):
                if isinstance(expr, Tensor):
                    return TensorAlias(expr, equations_map[expr.canonical_pyname()])
                children = expr.children()
                for i in range(len(children)):
                    children[i] = helper(children[i])
                return expr
            self.deepreturns = Return(*(helper(val)
                                      for val in self.returns[0].values))

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


class TensorAlias:
    def __init__(self, tensor, equation):
        self.tensor = tensor
        self.equation = equation

    def __repr__(self):
        return f"TensorAlias({self.tensor}, {self.equation})"

    def tensors(self):
        return [self.tensor] + self.equation.tensors()


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
        elif tree.data == 'compound_tensor':
            return CompoundTensor(*[run_tree(t) for t in tree.children])
        elif tree.data == 'index':
            name = run_tree(tree.children[0])
            if len(tree.children) > 1:
                idx = run_tree(tree.children[1]).value
            else:
                idx = None
            return Index(name, idx)
        elif tree.data == 'compound_index':
            return CompoundIndex([run_tree(t) for t in tree.children[:-1]], run_tree(tree.children[-1]))
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
        raise TypeError(
            f"transpose permutation isn't a permutation of operand dimensions, got permutation {dims} for operand with shape {tensor.shape}.")
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
            at_least_one_expansion = True
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
            collapse_later = [index for index in set(
                eq.indices) if index in outer_indices]
            return immediately_collapsable_str, collapse_later, {'is_tensor': False, 'collapsable': True}
        elif symbols[type(eq)]['precedence'] > prev_precedence:
            s, output_index, hret = helper(
                eq, symbols[type(eq)]['precedence'], outer_indices)
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
            child_cumulative_indices = [
                get_all_indices(child) for child in eq.children()]
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
                        outer_and_other_child_indices.update(
                            get_all_indices(child2))
                child_string, child_output_index, hdict = helper(
                    child, symbols[type(eq)]['precedence'], outer_and_other_child_indices)
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
            output_index = [index for index in outer_indices if index in output_index] + \
                           [index for index in output_index if index not in outer_indices]
            all_child_output_indices = set()
            for child_output_indices_ in child_output_indices:
                all_child_output_indices.update(child_output_indices_)
            input_and_output_indices = all_child_output_indices.union(
                output_index)
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
            einstring_right = "".join(
                index_letters[index] for index in output_index)
            einstring = ",".join(s for s in einstring_left) + \
                "->" + einstring_right
            return f"{''.join('(' + s + ')*' for s in nontensor_child_strings)}jnp.einsum('{einstring}', {', '.join(tensor_child_strings)})", output_index, {'is_tensor': is_tensor, 'collapsable': collapsable}
        elif isinstance(eq, (Sum, Power)):
            child_strings = []
            child_output_indices = []
            output_index = []
            is_tensor = False
            collapsable = False
            for child in eq.children():
                child_string, child_output_index, hdict = helper(
                    child, symbols[type(eq)]['precedence'], outer_indices)
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
                child_string, child_output_index, hdict = helper(
                    child, symbols[type(eq)]['precedence'], outer_indices)
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
        raise ValueError(
            f"This should never happen; we should have handled all cases.")

    outer_indices = list(eq.lhs.nonleading_indices())
    rhs_string = helper(eq.rhs, math.inf, outer_indices)[0]
    # Apply reshapes specified by compound indices
    final_output_shape_str = []
    any_output_index_is_compound = any(isinstance(index, CompoundIndex) for index in output_tensor.nonleading_indices(expand_compounds=False))
    if any_output_index_is_compound:
        for index in output_tensor.nonleading_indices(expand_compounds=False):
            if isinstance(index, CompoundIndex):
                final_output_shape_str.append(index.out_index.basename())
                sizes_required.update(in_index.basename() for in_index in index.in_indices)
            else:
                final_output_shape_str.append(index.basename())
                sizes_required.update(in_index.basename() for in_index in index.in_indices)
        final_output_shape_str = ",".join(final_output_shape_str)
        rhs_string = f"(({rhs_string}).reshape({final_output_shape_str}))"
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
        input_indices.update([index.basename()
                             for index in tensor.nonleading_indices()])
    parameter_indices = set()
    for tensor in parameter_tensors:
        parameter_indices.update([index.basename()
                                 for index in tensor.nonleading_indices()])
    # parameter_indices -= input_indices
    return parameter_indices, input_indices

def construct_jax_module_from_equations(name, eqs, inputs, jit, vmap):
    parameter_tensors, input_tensors = split_tensors(eqs, inputs)
    parameter_indices, input_indices = split_indices(parameter_tensors, input_tensors)
    # presized_input_indices = get_input_indices_that_do_not_depend_on_parameters(parameter_indices, input_indices)
    # init_indices = list(parameter_indices) + presized_input_indices
    parameter_arg_list = ', '.join(index for index in parameter_indices)
    parameter_kwarg_list = ''.join(f"{index}={index}, " for index in parameter_indices)
    s = f"class {name}:\n"
    s += f"    def __init__(self, {parameter_arg_list}):\n"
    dim_sizes = ', '.join(f"'{index}': {index}" for index in parameter_indices)
    s += f"        self.old_setup = self.setup\n"
    s += f"        self.setup = lambda *args, **kwargs: self.old_setup(*args, {parameter_kwarg_list}**kwargs)\n"
    s += "\n"
    s += "    " + \
        construct_jax_init_from_equations(eqs, inputs).replace("\n", "\n    ")
    s += "\n"
    s += "    @staticmethod\n"
    s += "    " + construct_jax_apply_from_equations(eqs=eqs, inputs=inputs, jit=jit, vmap=vmap).replace("\n", "\n    ")
    return s

def get_input_indices_that_do_not_depend_on_parameters(parameter_indices, input_indices):
    return [index for index in input_indices if index not in parameter_indices]

def construct_jax_init_from_equations(eqs, inputs):
    parameter_tensors, input_tensors = split_tensors(eqs, inputs)
    parameter_indices, input_indices = split_indices(parameter_tensors, input_tensors)
    # presized_input_indices = get_input_indices_that_do_not_depend_on_parameters(parameter_indices, input_indices)
    # init_indices = list(parameter_indices) + presized_input_indices
    s = ""
    s += "@staticmethod\n"
    s += f"def setup(key, {''.join(index + ', ' for index in parameter_indices)}**kwargs):\n"
    s += f"    keys = jax.random.split(key, {len(parameter_tensors)})\n"
    longest_name = max(len(tensor.canonical_pyname())
                       for tensor in parameter_tensors)
    tensor_defs = []
    for i, tensor in enumerate(parameter_tensors):
        shape_string = "[" + ', '.join(index.basename()
                                       for index in tensor.nonleading_indices()) + "]"
        tensor_defs.append({
            "name": tensor.canonical_pyname(),
            "key_name": f'"{tensor.canonical_pyname()}": ',
            "value": f"jax.random.normal(keys[{i}], shape={shape_string})"})
    max_name_len = max(len(tensor_def['key_name'])
                       for tensor_def in tensor_defs)
    param_dict_inner_str = "        " + ",\n        ".join(tensor_def['key_name'].ljust(
        max_name_len) + tensor_def['value'] for tensor_def in tensor_defs)
    s += f"    return {{\n{param_dict_inner_str}\n    }}\n"
    return s

def construct_jax_apply_from_equations(eqs, inputs, jit=True, vmap=True):
    parameter_tensors, input_tensors = split_tensors(eqs, inputs)
    parameter_indices, input_indices = split_indices(
        parameter_tensors, input_tensors)
    input_tensor_names = [tensor.pystr() for tensor in input_tensors]
    parameter_tensor_names = [tensor.pystr() for tensor in parameter_tensors]
    s = ""
    # s +=f"@lambda apply: vmap(apply, in_axes=({', '.join(['0' for _ in input_tensors] + ['None' for _ in parameter_tensors])}))\n"
    s += "@jit\n" if jit else ""
    s += f"@lambda func: vmap(func, in_axes=({', '.join(['0' for _ in input_tensors])}, None))\n" if vmap else ""
    s += f"def __call__({''.join([name + ', ' for name in input_tensor_names])}params):\n"
    # Indent
    sizes_required = set()
    eq_string = construct_equations(
        eqs, parameter_tensors, sizes_required=sizes_required).replace("\n", "\n    ")
    if len(sizes_required) > 0:
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
                        sizes.append(
                            f"params['{tensor.canonical_pyname()}'].shape[{i}]")
                        break_flag = True
                        break
                if break_flag:
                    break
            if not break_flag:
                raise ValueError(
                    f"Could not find index {index_basename} in {input_tensors + parameter_tensors}")
        s += "    " + ", ".join(sizes_required) + \
            " = " + ", ".join(sizes) + "\n"
    s += "    " + eq_string + "\n"
    s += "    " + eqs.returns[0].pystr()
    return s

def split_equations_along_index(eqs, split_index):
    subgraphs = []
    cache = {}

    def helper(e, inputs, outputs, prev_has_index):
        if e in cache:
            return cache[e]['inputs'].extend(inputs), cache[e]['outputs'].extend(outputs)
        if isinstance(e, Tensor) and (split_index in (i.basename() for i in e.nonleading_indices()) != prev_has_index):
            cache[e] = {'inputs': inputs, 'outputs': outputs}


def construct_jax_modules(body, main_name="main", main_inputs=None, jit=True, vmap=True):
    s = "import jax\nimport jax.numpy as jnp\nfrom jax import jit, vmap\n"
    ss = []
    functions = body.functions
    leftover_expressions = body
    for function in functions:
        ss.append(construct_jax_module_from_equations(function.name, function.body, function.arguments, jit=jit, vmap=vmap))
    s += '\n\n\n'.join(ss)
    s += construct_jax_module_from_equations(main_name, leftover_expressions, main_inputs if main_inputs is not None else [
    ], jit=jit) if leftover_expressions.returns else ""
    return s

# print(construct_jax_module_from_equations("TestModule", eqs, "x", False))


def jax_codegen(s, module_name="main", main_inputs=None, jit=True, vmap=True):
    tree = l.parse(s)
    eqs = run_tree(tree)
    c = construct_jax_modules(eqs, main_inputs, jit=jit, vmap=vmap)
    return c


def make_jax_module(s, jit=True, vmap=True):
    tree = l.parse(s)
    eqs = run_tree(tree)
    c = construct_jax_modules(eqs, jit=jit, vmap=vmap)
    exec(c)
    module_names = [function.name for function in eqs.functions]
    return eval(f"{', '.join(module_names)}")


def test_multiheadattention(jit=False, vmap=False):
    print("Running test_multiheadattention")
    einstein_code = """
    function MultiheadAttention(x)
        x[t,i] = x[t,i] * gx[i] * (1[i_2]/x[t,i_3]^2)^0.5
        q[h,t,j] = q[h,j,i] * x[t,i]
        k[h,t,j] = k[h,j,i] * x[t,i]
        v[h,t,j] = v[h,j,i] * x[t,i]
        a[h,t_1,t_2] = jnp.exp(q[h,t_1,j] * k[h,t_2,j])
        a[h,t_1,t_2] = a[h,t_1,t_2] / (a[h,t_3,t_2] + 0.000000001)
        u[t_1,h+hs=k] = wu[h,hs,j] * a[h,t_1,t_2] * v[h,t_2,j] + bu[k]
        z[t,i] = wz[k,i] * u[t,k]
        z[t,i] = z[t,i] * gz[i] * (1[i_2]/z[t,i_3]^2)^0.5
        return z[t,i]
    end
    """
    def activation(x):
        return jnp.maximum(x, 0)

    jax_code = jax_codegen(einstein_code, jit=jit, vmap=vmap)
    print(jax_code)

    # Initialise tensor shapes randomly
    key = jax.random.PRNGKey(0)
    for i in range(10):
        key, subkey = jax.random.split(key)
        i, hs, h, j, t = jax.random.randint(subkey, (5,), minval=1, maxval=10)
        k = h*hs
        print(f"i={i}, hs={hs}, k={k}, h={h}, j={j}, t={t}")
        def multiheadselfattention_reference(x_t_i, gx_i, q_h_j_i, k_h_j_i, v_h_j_i, wu_h_hs_j, bu_k, wz_k_i, gz_i):
            x_t_i = x_t_i * gx_i * (i/(x_t_i**2).sum(axis=-1, keepdims=True))**0.5
            q_h_t_j = jnp.einsum("ti,hji->htj", x_t_i, q_h_j_i)
            k_h_t_j = jnp.einsum("ti,hji->htj", x_t_i, q_h_j_i)
            v_h_t_j = jnp.einsum("ti,hji->htj", x_t_i, q_h_j_i)
            assert q_h_t_j.shape == (h, t, j)
            a_h_t_t = jnp.exp(jnp.einsum("haj,hbj->hab", q_h_t_j, k_h_t_j))
            a_h_t_t = a_h_t_t / (a_h_t_t.sum(axis=-1, keepdims=True) + 1e-9)
            assert a_h_t_t.shape == (h, t, t)
            u_t_k = jnp.reshape(jnp.einsum('hsj,hat,htj->hsa', wu_h_hs_j, a_h_t_t, v_h_t_j), [t,k]) + bu_k
            assert u_t_k.shape == (t, k)
            z_t_i = jnp.einsum("tk,ki->ti", u_t_k, wz_k_i)
            z_t_i = z_t_i * gz_i * (i/(z_t_i**2).sum(axis=-1, keepdims=True))**0.5
            assert z_t_i.shape == (t,i)
            return z_t_i

        
        MultiheadAttention = make_jax_module(einstein_code, jit=jit, vmap=vmap)
        atten = MultiheadAttention(i=i, hs=hs, k=k, h=h, j=j)
        key, subkey = jax.random.split(key)
        params = atten.setup(subkey)
        key, subkey = jax.random.split(key)
        x_t_i = jax.random.normal(subkey, (t, i))
        # print("Shapes:")
        # print(f"x_t_i: {x_t_i.shape}")
        # for name, param in params.items():
        #     print(f"{name}: {param.shape}")
        ref_result = multiheadselfattention_reference(x_t_i, **params)
        ein_result = atten(x_t_i, params)
        assert jnp.allclose(ein_result, ref_result)


if __name__ == "__main__":
    test_multiheadattention()