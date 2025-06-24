import streamlit as st
import sympy as sp
import numpy as np
import matplotlib.pyplot as plt

st.title("SymPy: Finding Derivatives at a Point")

# Example 1: f(x) = x**2
st.subheader("Example 1 ")

x = sp.symbols("x")

f = x**2
f_prime = sp.diff(f, x)
df = f.subs(x, 3)

st.latex("f(x) = " + sp.latex(f))

code_str = '''x = sp.symbols("x")

f = x**2

f_prime = sp.diff(f, x)

df = f.subs(x,3)'''

st.code(code_str, language="python")

st.latex("f'(x) = " + sp.latex(f_prime))
st.latex("f'(3) = " + sp.latex(f_prime.subs(x, 3)))
st.latex(f"f'(3) = {float(f_prime.subs(x, 3)):.4f}")

a = st.slider("Choose the x-coordinate of the point (a)", -6.0, 6.0, 3.0, step=0.1)
fa = a**2         # f(a)
fpa = 2*a         # f'(a)

x_vals = np.linspace(-6, 6, 400)
y_vals = x_vals**2
tangent_vals = fpa * (x_vals - a) + fa

fig, ax = plt.subplots()
ax.plot(x_vals, y_vals, label=r"$f(x) = x^2$")
ax.plot(x_vals, tangent_vals, '--', color='red', label=fr"Tangent at $x={a:.1f}$")
ax.scatter([a], [fa], color='black', zorder=5, label=fr"Point $({a:.1f}, {fa:.2f})$")
ax.axhline(0, color='gray', linewidth=0.5)
ax.axvline(0, color='gray', linewidth=0.5)
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_xlim(-6, 6)
ax.set_ylim(0, 40)
ax.text(a, fa + 5, fr"$f'({a:.1f}) = {fpa:.2f}$", color="red", fontsize=12, ha="center")
ax.legend()
st.pyplot(fig)

# Example 2: f(x) = sin(x)
st.subheader("Example 2")

f = sp.sin(x)
f_prime = sp.diff(f, x)
df = f.subs(x, 1)

st.latex("f(x) = " + sp.latex(f))

code_str = '''x = sp.symbols("x")

f = sp.sin(x)

f_prime = sp.diff(f, x)

df = f.subs(x, 1)'''

st.code(code_str, language="python")

st.latex("f'(x) = " + sp.latex(f_prime))
st.latex("f'(1) = " + sp.latex(f_prime.subs(x, 1)))
st.latex(f"f'(1) = {float(f_prime.subs(x, 1)):.4f}")

b = st.slider("Choose the x-coordinate of the point (b)", -2 * np.pi, 2 * np.pi, 0.0, step=0.1)
fb = np.sin(b)         # f(b)
fpb = np.cos(b)        # f'(b)

x_vals2 = np.linspace(-2 * np.pi, 2 * np.pi, 400)
y_vals2 = np.sin(x_vals2)
tangent_vals2 = fpb * (x_vals2 - b) + fb

fig2, ax2 = plt.subplots()
ax2.plot(x_vals2, y_vals2, label=r"$f(x) = \sin(x)$")
ax2.plot(x_vals2, tangent_vals2, '--', color='red', label=fr"Tangent at $x={b:.2f}$")
ax2.scatter([b], [fb], color='black', zorder=5, label=fr"Point $({b:.2f}, {fb:.2f})$")
ax2.axhline(0, color='gray', linewidth=0.5)
ax2.axvline(0, color='gray', linewidth=0.5)
ax2.set_xlabel("x")
ax2.set_ylabel("y")
ax2.set_xlim(-2 * np.pi, 2 * np.pi)
ax2.set_ylim(-2, 2)
ax2.text(b, fb + 0.5, fr"$f'({b:.2f}) = {fpb:.2f}$", color="red", fontsize=12, ha="center")
ax2.legend()
st.pyplot(fig2)

# Example 3: f(x) = x**x
st.subheader("Example 3")

f = x**x
f_prime = sp.diff(f, x)
df = f.subs(x, 2)

st.latex("f(x) = " + sp.latex(f))

code_str = '''x = sp.symbols("x")

f = x**x

f_prime = sp.diff(f, x)

df = f.subs(x, 2)'''

st.code(code_str, language="python")

st.latex("f'(x) = " + sp.latex(f_prime))
st.latex("f'(2) = " + sp.latex(f_prime.subs(x, 2)))
st.latex(f"f'(2) = {float(f_prime.subs(x, 2)):.4f}")

c = st.slider("Choose the x-coordinate of the point (c) for $f(x) = x^x$", 0.1, 4.0, 2.0, step=0.01)
fc = c**c
fpc = float(sp.diff(x**x, x).subs(x, c))

x_vals3 = np.linspace(0.1, 4, 400)
y_vals3 = x_vals3**x_vals3
tangent_vals3 = fpc * (x_vals3 - c) + fc

fig3, ax3 = plt.subplots()
ax3.plot(x_vals3, y_vals3, label=r"$f(x) = x^x$")
ax3.plot(x_vals3, tangent_vals3, '--', color='red', label=fr"Tangent at $x={c:.2f}$")
ax3.scatter([c], [fc], color='black', zorder=5, label=fr"Point $({c:.2f}, {fc:.2f})$")
ax3.axhline(0, color='gray', linewidth=0.5)
ax3.axvline(0, color='gray', linewidth=0.5)
ax3.set_xlabel("x")
ax3.set_ylabel("y")
ax3.set_xlim(0.1, 4)
ax3.set_ylim(0, np.max(y_vals3) + 2)
ax3.text(c, fc + 1, fr"$f'({c:.2f}) = {fpc:.2f}$", color="red", fontsize=12, ha="center")
ax3.legend()
st.pyplot(fig3)

# Example 4: f(x) = exp(-x**2)
st.subheader("Example 4")

f = sp.exp(-x**2)
f_prime = sp.diff(f, x)
df = f.subs(x, 1)

st.latex("f(x) = " + sp.latex(f))

code_str = '''x = sp.symbols("x")

f = sp.exp(-x**2)

f_prime = sp.diff(f, x)

df = f.subs(x, 1)'''

st.code(code_str, language="python")

st.latex("f'(x) = " + sp.latex(f_prime))
st.latex("f'(1) = " + sp.latex(f_prime.subs(x, 1)))
st.latex(f"f'(1) = {float(f_prime.subs(x, 1)):.4f}")

d = st.slider("Choose the x-coordinate of the point (d) for $f(x) = e^{-x^2}$", -2.0, 2.0, 1.0, step=0.01)
fd = np.exp(-d**2)
fpd = float(sp.diff(sp.exp(-x**2), x).subs(x, d))

x_vals4 = np.linspace(-2, 2, 400)
y_vals4 = np.exp(-x_vals4**2)
tangent_vals4 = fpd * (x_vals4 - d) + fd

fig4, ax4 = plt.subplots()
ax4.plot(x_vals4, y_vals4, label=r"$f(x) = e^{-x^2}$")
ax4.plot(x_vals4, tangent_vals4, '--', color='red', label=fr"Tangent at $x={d:.2f}$")
ax4.scatter([d], [fd], color='black', zorder=5, label=fr"Point $({d:.2f}, {fd:.2f})$")
ax4.axhline(0, color='gray', linewidth=0.5)
ax4.axvline(0, color='gray', linewidth=0.5)
ax4.set_xlabel("x")
ax4.set_ylabel("y")
ax4.set_xlim(-2, 2)
ax4.set_ylim(-0.1, 1.1)
ax4.text(d, fd + 0.1, fr"$f'({d:.2f}) = {fpd:.2f}$", color="red", fontsize=12, ha="center")
ax4.legend()
st.pyplot(fig4)

st.header("Homework: Compute Derivatives at a Point")

st.markdown("""
Try the following on your own. For each function, compute the derivative at the given point.  
Write your answer as a decimal rounded to four decimal places.
""")

st.markdown(r"""
$1.\quad f(x) = \ln(x^2 + 1)\quad \text{at}\ x = 2$  
$2.\quad f(x) = \cos(x)\, e^x\quad \text{at}\ x = 0$  
$3.\quad f(x) = \dfrac{1}{x^2 + 1}\quad \text{at}\ x = 1$  
$4.\quad f(x) = x^3 - 5x + 4\quad \text{at}\ x = -1$  
$5.\quad f(x) = x \sin(x)\quad \text{at}\ x = \pi$  
""")

st.markdown("**Solutions:**")

# Solution 1
x = sp.symbols("x")
f1 = sp.log(x**2 + 1)
f1_prime = sp.diff(f1, x)
sol1 = float(f1_prime.subs(x, 2))

# Solution 2
f2 = sp.cos(x) * sp.exp(x)
f2_prime = sp.diff(f2, x)
sol2 = float(f2_prime.subs(x, 0))

# Solution 3
f3 = 1 / (x**2 + 1)
f3_prime = sp.diff(f3, x)
sol3 = float(f3_prime.subs(x, 1))

# Solution 4
f4 = x**3 - 5*x + 4
f4_prime = sp.diff(f4, x)
sol4 = float(f4_prime.subs(x, -1))

# Solution 5
f5 = x * sp.sin(x)
f5_prime = sp.diff(f5, x)
sol5 = float(f5_prime.subs(x, np.pi))

st.markdown(
    f"""
$1.\\quad f'(2) = {sol1:.4f}$  
$2.\\quad f'(0) = {sol2:.4f}$  
$3.\\quad f'(1) = {sol3:.4f}$  
$4.\\quad f'(-1) = {sol4:.4f}$  
$5.\\quad f'(\\pi) = {sol5:.4f}$
"""
)