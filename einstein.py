"""
A small library for specifying neural networks using index notation similar to Einstein summation notation.
"""

import numpy as np
import jax.numpy as jnp
import uuid
from IPython.display import display, Math


class IndexExpression:
    """
    A class for representing index expressions
    """
    def __init__(self, operation, subexpressions):
        self.operation = operation
        self.subexpressions = subexpressions
        
    def __eq__(self, other):
        return self.operation == other.operation and self.subexpressions == other.subexpressions
    
    def __hash__(self):
        return hash(str(self))

    def __str__(self):
        return str(self.operation) + '(' + ','.join(str(s) for s in self.subexpressions) + ')'

    def __repr__(self):
        return str(self)
    
    def __add__(self, other):
        return IndexExpression('+', [self, other])
    
    def __radd__(self, other):
        return IndexExpression('+', [other, self])
    
    def __sub__(self, other):
        return IndexExpression('-', [self, other])
    
    def __rsub__(self, other):
        return IndexExpression('-', [other, self])
    
    def __mul__(self, other):
        return IndexExpression('*', [self, other])
    
    def __rmul__(self, other):
        return IndexExpression('*', [other, self])
    
    def __truediv__(self, other):
        return IndexExpression('/', [self, other])
    
    def __rtruediv__(self, other):
        return IndexExpression('/', [other, self])
    
    def __pow__(self, other):
        return IndexExpression('**', [self, other])
    
    def __rpow__(self, other):
        return IndexExpression('**', [other, self])

    def __neg__(self):
        return IndexExpression('-', [self])
    
    def __pos__(self):
        return IndexExpression('+', [self])
    
    def __abs__(self):
        return IndexExpression('abs', [self])
    
    def __invert__(self):
        return IndexExpression('~', [self])
    
    def __call__(self, *args):
        return IndexExpression('()', [self] + list(args))
        
    def __lt__(self, other):
        return IndexExpression('<', [self, other])
            
    def __le__(self, other):
        
        return IndexExpression('<=', [self, other])
    
    def __gt__(self, other):
        return IndexExpression('>', [self, other])
    
    def __ge__(self, other):
        return IndexExpression('>=', [self, other])
    
    def __eq__(self, other):
        return IndexExpression('==', [self, other])
    
    def __ne__(self, other):
        return IndexExpression('!=', [self, other])
    
    def __and__(self, other):
        return IndexExpression('&', [self, other])
    
    def __or__(self, other):
        return IndexExpression('|', [self, other])
    
    def __xor__(self, other):
        return IndexExpression('^', [self, other])
        
    def __getstate__(self):
        return self.operation, self.subexpressions
    
    def __setstate__(self, state):
        self.operation, self.subexpressions = state
        
    def to_latex(self, prev_precedence=0):
        """
        Convert the expression to LaTeX
        """
        precedence = {
            '==': 1,
            '!=': 1,
            '<': 2,
            '<=': 2,
            '>': 2,
            '>=': 2,
            '+': 3,
            '-': 3,
            '*': 4,
            '/': 0,
            '**': 5,
            '&': 6,
            '|': 6,
            '^': 0,
            '~': 6,
            '()': 7,
            'abs': -1,
        }
        operators = {
            '+': ['', '+', ''],
            '-': ['', '-', ''],
            '*': ['', ' \\cdot ', ''],
            '/': ['\\frac{', '}{', '}'],
            '**': ['', '^{', '}'],
            '<': ['', '<', ''],
            '<=': ['', ' \\leq ', ''],
            '>': ['', '>', ''],
            '>=': ['', ' \\geq ', ''],
            '==': ['', '=', ''],
            '!=': ['', ' \\neq ', ''],
            '&': ['', ' \\wedge ', ''],
            '|': ['', ' \\vee ', ''],
            '^': ['', ' \\oplus ', ''],
            '()': ['(', ')', ''],
            '[]': ['[', ']', ''],
            'abs': ['\\left|', '', '\\right|'],
            '~': ['\\left\\lnot', '', ''],
            '<>': ['', ' \\neq ', ''],
            '<': ['', '<', ''],
            '>': ['', '>', ''],
            '<=': ['', ' \\leq ', ''],
            '>=': ['', ' \\geq ', ''],
            '==': ['', '=', ''],
            '!=': ['', ' \\neq ', ''],
        }
        
        if isinstance(self, IndexExpression):
            s = ''
            for i, subexpression in enumerate(self.subexpressions):
                if i > 0:
                    s += operators[self.operation][1]
                if isinstance(subexpression, IndexExpression):
                    s += subexpression.to_latex(precedence[self.operation])
                else:
                    s += str(subexpression)
            s = operators[self.operation][0] + s + operators[self.operation][2]
            # If this operator (e.g. *) binds more tightly than its parent (e.g. +), parenthesis are not needed. Otherwise, they are.
            if precedence[self.operation] == -1:
                pass
            elif precedence[self.operation] < prev_precedence:
                s = '\\left(' + s + '\\right)'
            return s
        else:
            return str(self)
        
        
                
                
class Indices(IndexExpression):
    """
    A class for representing expressions involving indices
    """
    def __init__(self, indices):
        self.indices = indices
        self.realised = False

    def __str__(self):
        return ','.join(str(i) for i in self.indices)

    def __repr__(self):
        return str(self)
    
    def to_latex(self, prev_precedence=0):
        s = self.indices[0] + r"_{"
        for i, index in enumerate(self.indices[1:]):
            if isinstance(index, tuple):
                s += str(index[0]) + r"_{" + str(index[1]) + r"}"
            else:
                s += str(index)
            if i < len(self.indices) - 2:
                s += r" "
        s += r"}"
        return s
    
    def __eq__(self, other):
        return self.indices == other.indices

    def __hash__(self):
        return hash(str(self))
    

class Size(IndexExpression):
    """
    A class for representing expressions involving sizes
    """
    def __init__(self, name):
        self.name = name

    def __str__(self):
        return '||' + self.name + '||'

    def __repr__(self):
        return str(self)
    
    def to_latex(self, prev_precedence=0):
        return r"\|" + self.name + r"\|"

    def __hash__(self):
        return hash(str(self))


class Realised(IndexExpression):
        """
        Represents realised values of an index. For example, $x(t) = x_0 e^{-t}$ would be represented as `ie['x','t'] = ie['x_0',{'name':'t', 'value':0}] * np.euler**(-ie['t']).value()`
        """
        def __init__(self, name):
            self.name = name
            self.realised = True

        def __str__(self):
            return str(self.name)

        def __repr__(self):
            return str(self)
        
        def to_latex(self, prev_precedence=0):
            return str(self.name)

        def __hash__(self):
            return hash(str(self))


class Concatenate(IndexExpression):
    """
    A class for representing concatenation
    """
    def __init__(self, from_indices, to_indices):
        self.from_indices = from_indices
        self.to_indices = to_indices
        self.realised = False

    def __str__(self):
        return ','.join(str(i) for i in self.from_indices) + '->' + ','.join(str(i) for i in self.to_indices)

    def __repr__(self):
        return str(self)
    
    def to_latex(self, prev_precedence=0):
        # Concatenation operator: $\mathbin{+\mkern-10mu+}_{abc \to d}
        return r"\mathbin{+\mkern-10mu+}_{{" + ' '.join(i[0] + r"_{" + str(i[1]) + r"}" if isinstance(i, tuple) else str(i) for i in self.from_indices.indices) + r"} \to {" + ' '.join(i[0] + r"_{" + str(i[1]) + r"}" if isinstance(i, tuple) else str(i) for i in self.to_indices.indices) + r"}}"
    
    def __eq__(self, other):
        return self.from_ == other.from_ and self.to_ == other.to_

    def __hash__(self):
        return hash(str(self))
    
    
class Function(IndexExpression):
    """
    Represents a function
    """
    def __init__(self, name, function=None):
        self.name = name
        self.function = function
        self.parameters = None
                
    def __str__(self):
        return self.name + '(' + ','.join(str(i) for i in self.parameters) + ')'
    
    def __repr__(self):
        return str(self)

    def to_latex(self, prev_precedence=0):
        return r"\mathrm{" + self.name + r"}" + r"\left(" + ','.join(p.to_latex() for p in self.parameters) + r"\right)"
    
    def __eq__(self, other):
        return self.name == other.name and self.parameters == other.parameters
    
    def __hash__(self):
        return hash(str(self))
    
    def __call__(self, *args):
        self.parameters = args
        return self
        

class IndexEquations:
    """
    Represents equations in index notation. In essence, it assigns each sequence of indices to an expression of indices.
    """
    def __init__(self):
        self.equations = {}

    def __setitem__(self, names, value):
        """
        Define a variable as a sequence of indices.
        """
        if isinstance(names, str):
            names = tuple([s.strip() for s in names.split(",") if s.strip() != ""])
        if isinstance(names, tuple):
            indices = Indices(names)
        if isinstance(names, Indices):
            indices = names
        self.equations[indices] = value

    def __getitem__(self, names):
        """
        Get an index representation of a variable as a sequence of indices.
        """
        if isinstance(names, str):
            names = tuple([s.strip() for s in names.split(",") if s.strip() != ""])
        return Indices(names)
    
    def get_realisation(self, name):
        """
        Get a representation of a realisation of a variable.
        """
        return Indices([name]).value()
    
    def get_size(self, name):
        """
        Get the size of a variable.
        """
        return Size(name)

    def __str__(self):
        s = "\\begin{align*}\n"
        for k, v in self.equations.items():
            s += k.to_latex() + " &= " + v.to_latex() + " \\\\\n"
        s += "\\end{align*}"
        return s
    
"""
$$
\begin{align*}
q_{b h t c} &= t \\
k_{b h t c} &= t \\
v_{b h t c} &= x_{b t c} \\
a_{b h t_{0} t_{1}} &= \left|k_{t_{0} c}-k_{t_{1} c}\right| \leq 1 \\
u^2_{b t_{0} c} &= U_{c_{0} c_{1}} \cdot a_{h t_{0} t_{1}} \cdot v_{h t_{1} c_{0}} \\
u^1_{b t c} &= x_{b t c}+u^1_{b t c} \\
u^0_{b t c} &= \left(\frac{\gamma_{c} \cdot \left(u^1_{b t c}-\mu_{c}\right)}{\sigma_{c}}\right)+\beta_{c} \\
\mu_{c} &= \left(\frac{1_{b t}}{\|b\| \cdot \|t\|}\right) \cdot u^1_{b t c} \\
\sigma_{c} &= \left(\left(\frac{1_{b t}}{\|b\| \cdot \|t\|}\right) \cdot \left(u^1_{b t c}-\mu_{c}\right)^{2}\right)^{0.5} \\
z_{b c} &= \left(\frac{\gamma_{c} \cdot \left(z_{b c}-\mu_{c}\right)}{\sigma_{c}}\right)+\beta_{c} \\
\end{align*}
$$
"""

k = 1
ie = IndexEquations()
v = Realised
s = Size
ie['q,b,h,t,c'] = v('t')
ie['k,b,h,t,c'] = v('t')
ie['v,b,h,t,c'] = ie['x,b,t,c']
ie['a','b','h',('t',0), ('t',1)] = abs(ie['k',('t',0),'c'] - ie['k',('t',1),'c']) <= k
ie["u^2",'b',('t',0),'c'] = ie['U',('c',0),('c',1)] * ie['a','h',('t',0),('t',1)] * ie['v','h',('t',1),('c',0)]
ie["u^1",'b','t','c'] = ie['x','b','t','c'] + ie["u^1",'b','t','c']
ie["u^0",'b','t','c'] = ie['\\gamma','c'] * (ie['u^1','b','t','c'] - ie['\\mu','c']) / ie['\\sigma','c'] + ie['\\beta','c']
ie['\\mu','c'] = ie['1','b','t'] / (s('b') * s('t')) * ie["u^1",'b','t','c']
ie['\\sigma','c'] = (ie['1','b','t'] / (s('b') * s('t')) * (ie["u^1",'b','t','c'] - ie['\\mu','c']) ** 2) ** 0.5
ie['z','b','c'] = ie['\\gamma','c'] * (ie['z','b','c'] - ie['\\mu','c']) / ie['\\sigma','c'] + ie['\\beta','c']

display(Math(str(ie)))
# print(str(ie))