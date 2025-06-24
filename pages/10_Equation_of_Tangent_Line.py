import streamlit as st
import sympy as sp
import numpy as np
import matplotlib.pyplot as plt

st.subheader("Example 1")

st.write(f"Find the equation of the line tangent to $f(x) = x^2$ where $x = 3$.")

# working it out for myself
x = sp.symbols('x')
f = x**2
f_prime = sp.diff(f, x)
df = f_prime.subs(x, 3)
st.latex(r"f(3) = 9 \;\;\longrightarrow\;\; (3, 9)")
st.latex(f"f'(x) = {sp.latex(f_prime)}")
st.latex(f"f'(3) = {sp.latex(df)}")
st.write(f"The slope of the tangent line at $x = 3$ is $f'(3) = {sp.latex(df)}$.")



# Show solving for y step by step, left-aligned by the first symbol (with extra spacing)
st.latex(r"""
\begin{array}{l}
6 = \dfrac{y - 9}{x - 3} \\[1.2em]
6(x - 3) = y - 9 \\[1.2em]
y = 6(x - 3) + 9 \\[1.2em]
y = 6x - 18 + 9 \\[1.2em]
y = 6x - 9
\end{array}
""")

# Plot f(x) = x^2 and its tangent at x = 3
x_vals = np.linspace(0, 6, 400)
y_vals = x_vals**2
tangent_vals = 6 * (x_vals - 3) + 9  # slope 6, passes through (3,9)

fig, ax = plt.subplots()
ax.plot(x_vals, y_vals, label=r"$f(x) = x^2$")
ax.plot(x_vals, tangent_vals, '--', color='red', label="Tangent at $x=3$")
ax.scatter([3], [9], color='black', zorder=5, label="Point $(3, 9)$")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_xlim(0, 6)
ax.set_ylim(0, 40)
ax.legend()
st.pyplot(fig)

# Example 2: f(x) = sin(x) at x = pi/4
st.subheader("Example 2")

st.write(f"Find the equation of the line tangent to $f(x) = \\sin(x)$ where $x = \\frac{{\\pi}}{{4}}$.")

x = sp.symbols('x')
f = sp.sin(x)
f_prime = sp.diff(f, x)
x0 = np.pi / 4
y0 = np.sin(x0)
slope = float(f_prime.subs(x, x0))
b_val = y0 - slope * x0  # <-- Add this line

# Build the LaTeX string first, then pass to st.latex
latex_point = rf"f\left(\frac{{\pi}}{{4}}\right) = {y0:.4f} \;\;\longrightarrow\;\; \left(\frac{{\pi}}{{4}},\; {y0:.4f}\right)"
st.latex(latex_point)
st.latex(f"f'(x) = {sp.latex(f_prime)}")
st.latex(rf"f'\left(\frac{{\pi}}{{4}}\right) = {slope:.4f}")
st.write(f"The slope of the tangent line at $x = \\frac{{\\pi}}{{4}}$ is $f'(\\frac{{\\pi}}{{4}}) = {slope:.4f}$.")

latex_steps2 = rf"""
\begin{{array}}{{l}}
{slope:.4f} = \dfrac{{y - {y0:.4f}}}{{x - \frac{{\pi}}{{4}}}} \\[1.2em]
{slope:.4f}(x - \frac{{\pi}}{{4}}) = y - {y0:.4f} \\[1.2em]
y = {slope:.4f}(x - \frac{{\pi}}{{4}}) + {y0:.4f} \\[1.2em]
y = {slope:.4f}x - {slope * x0:.4f} + {y0:.4f} \\[1.2em]
y = {slope:.4f}x + {b_val:.4f}
\end{{array}}
"""
st.latex(latex_steps2)

x_vals = np.linspace(0, np.pi, 400)
y_vals = np.sin(x_vals)
tangent_vals = slope * (x_vals - x0) + y0

fig, ax = plt.subplots()
ax.plot(x_vals, y_vals, label=r"$f(x) = \sin(x)$")
ax.plot(x_vals, tangent_vals, '--', color='red', label=r"Tangent at $x=\frac{\pi}{4}$")
ax.scatter([x0], [y0], color='black', zorder=5, label=fr"Point $(\frac{{\pi}}{{4}},\; {y0:.2f})$")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_xlim(0, np.pi)
ax.set_ylim(-0.2, 1.2)
ax.legend()
st.pyplot(fig)

# Example 3: f(x) = ln(x) at x = 2
st.subheader("Example 3")

st.write(f"Find the equation of the line tangent to $f(x) = \\ln(x)$ where $x = 2$.")

f = sp.log(x)
f_prime = sp.diff(f, x)
x0 = 2
y0 = np.log(x0)
slope = float(f_prime.subs(x, x0))
st.latex(r"f(2) = " + f"{y0:.4f} \;\;\longrightarrow\;\; (2,\; {y0:.4f})")
st.latex(f"f'(x) = {sp.latex(f_prime)}")
st.latex(f"f'(2) = {slope:.4f}")
st.write(f"The slope of the tangent line at $x = 2$ is $f'(2) = {slope:.4f}$.")

# Continue algebraic steps to slope-intercept form for Example 3
b_val = y0 - slope * x0
latex_steps3 = rf"""
\begin{{array}}{{l}}
{slope:.4f} = \dfrac{{y - {y0:.4f}}}{{x - 2}} \\[1.2em]
{slope:.4f}(x - 2) = y - {y0:.4f} \\[1.2em]
y = {slope:.4f}(x - 2) + {y0:.4f} \\[1.2em]
y = {slope:.4f}x - {slope * x0:.4f} + {y0:.4f} \\[1.2em]
y = {slope:.4f}x + {b_val:.4f}
\end{{array}}
"""
st.latex(latex_steps3)

x_vals = np.linspace(0.5, 3, 400)
y_vals = np.log(x_vals)
tangent_vals = slope * (x_vals - x0) + y0

fig, ax = plt.subplots()
ax.plot(x_vals, y_vals, label=r"$f(x) = \ln(x)$")
ax.plot(x_vals, tangent_vals, '--', color='red', label="Tangent at $x=2$")
ax.scatter([x0], [y0], color='black', zorder=5, label=fr"Point $(2,\; {y0:.2f})$")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_xlim(0.5, 3)
ax.set_ylim(np.log(0.5)-0.5, np.log(3)+0.5)
ax.legend()
st.pyplot(fig)

st.header("Homework: Find the Equation of the Tangent Line")

st.markdown("""
For each function and value of $x$, find the equation of the tangent line in slope-intercept form $y = mx + b$.
""")

homework = [
    (r"f(x) = x^3", 1),
    (r"f(x) = e^x", 0),
    (r"f(x) = \ln(x)", 1),
    (r"f(x) = \cos(x)", 0),
    (r"f(x) = x^2 + 2x", -1),
]

x = sp.symbols('x')
solutions = []

for idx, (f_latex, x_val) in enumerate(homework, 1):
    # Parse the function string to sympy
    if "x^3" in f_latex:
        f = x**3
    elif "e^x" in f_latex:
        f = sp.exp(x)
    elif "ln(x)" in f_latex:
        f = sp.log(x)
    elif "cos(x)" in f_latex:
        f = sp.cos(x)
    elif "x^2 + 2x" in f_latex:
        f = x**2 + 2*x
    else:
        continue

    f_prime = sp.diff(f, x)
    slope = float(f_prime.subs(x, x_val))
    y0 = float(f.subs(x, x_val))
    b_val = y0 - slope * x_val

    # Display problem
    st.latex(rf"{idx}.\quad f(x) = {f_latex},\quad x = {x_val}")

    # Display solution
    st.markdown("**Solution:**")
    st.latex(rf"f'(x) = {sp.latex(f_prime)}")
    st.latex(rf"f'({x_val}) = {slope:.4f}")
    st.latex(rf"f({x_val}) = {y0:.4f}")
    st.latex(rf"y = {slope:.4f}x + {b_val:.4f}")


