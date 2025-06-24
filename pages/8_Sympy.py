import streamlit as st
import sympy as sp

st.title("SymPy: Finding Derivatives")

x = sp.symbols("x")

examples = [
    ("f = x**2", x**2),
    ("f = sp.sin(x)", sp.sin(x)),
    ("f = sp.exp(x)", sp.exp(x)),
    ("f = x**3 - 2*x + 7", x**3 - 2*x + 7),
    ("f = sp.log(x)", sp.log(x)),
    ("f = x**2 * sp.exp(x)", x**2 * sp.exp(x)),
    ("f = sp.cos(x) + x**2", sp.cos(x) + x**2),
    ("f = sp.sqrt(x)", sp.sqrt(x)),
    ("f = 1 / (x + 1)", 1 / (x + 1)),
    ("f = sp.sin(x) * sp.exp(x)", sp.sin(x) * sp.exp(x)),
    ("f = sp.exp(-x**2)", sp.exp(-x**2)),
    ("f = x**x", x**x),
    ("f = sp.log(x**2 + 1)", sp.log(x**2 + 1)),
]

for i, (code_str, f) in enumerate(examples, 1):
    st.subheader(f"Example {i}")
    f_prime = sp.diff(f, x)
    st.latex("f(x) = " + sp.latex(f))
    code = f'''
x = sp.symbols("x")
{code_str}
f_prime = sp.diff(f, x)
'''
    st.code(code, language="python")
    st.latex("f'(x) = " + sp.latex(f_prime))








