# from lark import Lark, Transformer, v_args
from lark import Lark
import jax
from jax import numpy as jnp
from jax import random
from jax import jit, vmap, grad, jacobian

l = Lark.open("einstein.lark")
# l = Lark(r"""
# start: equation+

# equation: value "=" expression
# expression: sum
# sum: product (sum_term | sub_term)*
# sum_term: "+" product
# sub_term: "-" product
# product: factor (prod_term | div_term)*
# prod_term: "*" factor
# div_term: "/" factor
# factor: power | value
# power: value ("^" value)
# value   : IDENTIFIER "(" expression ")" -> function_call
#         | "(" expression ")"            -> paren_expr
#         | tensor                        -> tensor
#         | NUMBER
# tensor: index+
# index: LETTER ("_" NUMBER)?
# IDENTIFIER: LETTER (LETTER|NUMBER)*
# LETTER: "a".."z"
# NUMBER: "0".."9"+

# WHITESPACE: (" " | "\n")+
# %ignore WHITESPACE
# """)

text = """
zbi_0 = sigma(wi_1i_0^2 * xbi_0 + bi_0)
return zbi
"""

tree = l.parse(text)
print(tree.pretty())

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


class Equations:
    def __init__(self, *equations):
        self.equations = [equations for equations in equations if isinstance(equations, Equation)]
        self.returns = [equations for equations in equations if isinstance(equations, Return)]
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
            return Equations(*[run_tree(e) for e in tree.children])
        elif tree.data == 'statement':
            return run_tree(tree.children[0])
        elif tree.data == 'return':
            return Return(*[run_tree(e) for e in tree.children])
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
            return run_tree(tree.children[0])
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
            return run_tree(tree.children[0])
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
    elif tree.type == 'IDENTIFIER':
        return tree.value
    elif tree.type == 'LETTER':
        return tree.value
    elif tree.type == 'NUMBER':
        return Number(tree.value)
    raise ValueError(f"Unknown tree: {tree}")
    


eqs = run_tree(tree)
print(eqs)
# print(eqs.make_apply())
print(eqs.tensors())

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
    numbered_dims = ", ".join(str(dim) for dim in dims if dim is not None)
    index_str = ', '.join("None" if dim is None else ":" for dim in dims)
    return f"{tensor_name}.transpose({numbered_dims})[{index_str}]"

def codegen_collapse(tensor_name, dims):
    """
    >>> codegen_collapse("x", [1, None, 0])
    'x = np.sum(x, axis=1, keepdims=True)'
    """
    return f"{tensor_name}.sum(axis={dims})"

def construct_equation(eq):
    """
    >>> eq = Equation(Tensor(Index('y', None), Index('b', None), Index('i', None)), Product(Tensor(Index('w', None), Index('i', 1), Index('i', None)), Tensor(Index('x', None), Index('b', None), Index('i', 1))))
    >>> construct_equation(eq)
    'y_b_i = (w_i_i1.transpose(1, 0)[None, :, :] * x_b_i1.transpose(0, 1)[:, None, :]).sum(axis=(1,))'
    """
    transpand_dims, collapse_dims = get_transexpand_and_collapse_instructions(eq)
    
    symbols = {
        Sum: {'symbol': '+', 'is_pycall': False, 'precedence': 4},
        Product: {'symbol': '*', 'is_pycall': False, 'precedence': 3},
        Power: {'symbol': '**', 'is_pycall': False, 'precedence': 9},
        FunctionCall: {'symbol': '', 'is_pycall': True, 'precedence': 1},
    }
    def helper(e, prev_precedence):
        if isinstance(e, Number):
            return str(e.value)
        elif isinstance(e, Tensor):
            return codegen_transexpand(e.pystr(), transpand_dims[e])
        symbol_data = symbols[type(e)]
        if symbol_data['is_pycall']:
            return f"{e.name}({', '.join(helper(child, symbol_data['precedence']) for child in e.children())})"
        else:
            if prev_precedence > symbol_data['precedence']:
                return f"({helper(e, symbol_data['precedence'])})"
            else:
                return symbol_data['symbol'].join(helper(child, symbol_data['precedence']) for child in e.children())
    return f"{eq.lhs.pystr()} = {helper(eq.rhs, 0)}"

def construct_equations(eqs):
    return "\n".join(construct_equation(eq) for eq in eqs.equations)

def construct_jax_apply_from_equations(eqs, jit=False):
    s = "@jit\n" if jit else ""
    parameter_tensors = []
    defined_tensors = set()
    for eq in eqs.equations:
        for tensor in eq.rhs.tensors():
            tensor_str = tensor.canonical_pyname()
            if tensor_str not in defined_tensors:
                parameter_tensors.append(tensor_str)
            defined_tensors.add(tensor_str)
    s += f"def apply({', '.join(parameter_tensors)}):\n"
    # Indent
    s += "    " + construct_equations(eqs).replace("\n", "\n    ") + "\n"
    s += "    " + eqs.returns[0].pystr()
    return s

def construct_jax_init_from_equations(eqs, inputs):
    s = f"def init(key, dim_sizes):\n"
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
    tensors = parameter_tensors
    indices = set()
    for tensor in tensors:
        indices.update(index.basename() for index in tensor.nonleading_indices())
    s += f"    keys = jax.random.split(key, {len(tensors)})\n"
    longest_name = max(len(tensor.canonical_pyname()) for tensor in tensors)
    tensor_defs = []
    for i, tensor in enumerate(tensors):
        shape_string = "(dim_sizes[dim] for dim in [" + ', '.join(f"'{index.basename()}'" for index in tensor.nonleading_indices()) + "])"
        tensor_defs.append({"name": tensor.canonical_pyname().ljust(longest_name), "value": f"jax.random.normal(keys[{i}], shape={shape_string}))"})
    param_dict_inner_str = "        " + "\n        ".join(f'"{tensor_def["name"]}: {tensor_def["value"]}' for tensor_def in tensor_defs)
    s += f"    return {{\n{param_dict_inner_str}\n    }}\n"
    return s

def construct_jax_module_from_equations(name, eqs, inputs, jit):
    s = f"class {name}:\n"
    s += "    def __init__(self, dim_sizes):\n"
    s += "        self.dim_sizes = dim_sizes\n"
    s += "\n"
    s += "    " + construct_jax_init_from_equations(eqs, inputs).replace("\n", "\n    ")
    s += "\n"
    s += "    " + construct_jax_apply_from_equations(eqs, jit).replace("\n", "\n    ")
    return s


print(construct_jax_module_from_equations("TestModule", eqs, "x", False))




def print_tests():
    transexpand_dims, collapse_dims = get_transexpand_and_collapse_instructions(eqs.equations[0])
    for tensor, dims in transexpand_dims.items():
        print(f"{tensor.pystr()} -> {dims}")
    print(f"collapse final along axes {collapse_dims}")

    print(transexpand(jnp.array([[1, 2, 3],[4, 5, 6]]), [None, 1, None, None, 0]).shape)
    print(codegen_transexpand("x", [None, 1, None, None, 0]))
    print(codegen_collapse("x", [1, None, 0]))

    print(eqs.equations[0])
    print(construct_equation(eqs.equations[0]))


# print_tests()