import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
import io
import pandas as pd

st.title("Regression More Points, Fixed Line Through Origin")

# Generate 10 random points in the plane (for reproducibility, set a seed)
np.random.seed(42)
xs = np.linspace(0, 10, 10)
ys = 2 * xs + 1 + np.random.normal(0, 3, size=10)  # Some linear trend + noise

# Fixed value for a
a = 1.0

fig, ax = plt.subplots()
ax.scatter(xs, ys, color="red", label="Data points")
ax.plot(xs, a * xs, color="blue", label=fr"$y = {a:.1f}x$")
# Label only the first three red points, with the first label directly underneath to avoid overlap
for i in range(3):
    if i == 0:
        ax.text(xs[i], ys[i]-0.7, f"({xs[i]:.1f}, {ys[i]:.1f})", color="red", fontsize=7, va="top", ha="center")
    else:
        ax.text(xs[i]+0.1, ys[i], f"({xs[i]:.1f}, {ys[i]:.1f})", color="red", fontsize=7, va="bottom")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_xlim(-1, 11)
ax.set_ylim(min(ys)-5, max(ys)+5)
st.pyplot(fig)

# --- Symbolic MSE and Derivative, and Conversion to Python Expressions ---

# Symbolic variable for a
a_sym = sp.Symbol('a')

# Calculate MSE symbolically, point by point
mse_expr = 0
for x, y in zip(xs, ys):
    mse_expr += (a_sym * x - y)**2
mse_expr = mse_expr / len(xs)

# Derivative of MSE with respect to a
dmse_da = sp.diff(mse_expr, a_sym)

# Display the MSE in a vertical, multi-line form
n = len(ys)
mse_terms = [f"(a \\cdot {x:.2f} - {y:.2f})^2" for x, y in zip(xs, ys)]
align_lines = [f"\mathrm{{MSE}}(a) &= \\frac{{1}}{{{n}}} {mse_terms[0]}"] + [f"\\\\&\\quad+ \\frac{{1}}{{{n}}} {term}" for term in mse_terms[1:]]
mse_latex_vertical = (
    r"\begin{align*}"
    + "".join(align_lines)
    + r"\end{align*}"
)
st.markdown("**Symbolic Mean Squared Error (MSE) for these points:**")
st.latex(mse_latex_vertical)

# Display the derivative
dmse_terms = [f"(a \\cdot {x:.2f} - {y:.2f}) \\cdot {x:.2f}" for x, y in zip(xs, ys)]
align_lines_dmse = [f"\\frac{{d\\,\\mathrm{{MSE}}}}{{da}} &= \\frac{{2}}{{{n}}} {dmse_terms[0]}"] + [f"\\\\&\\quad+ \\frac{{2}}{{{n}}} {term}" for term in dmse_terms[1:]]
dmse_latex_vertical = (
    r"\begin{align*}"
    + "".join(align_lines_dmse)
    + r"\end{align*}"
)
st.markdown("**Derivative of the MSE with respect to $a$:**")
st.latex(dmse_latex_vertical)

# --- Convert symbolic expressions to Python functions ---
mse_func = sp.lambdify(a_sym, mse_expr, modules='numpy')
dmse_da_func = sp.lambdify(a_sym, dmse_da, modules='numpy')

st.markdown("**Python code to create the symbolic MSE, its derivative, and convert them to Python functions:**")
st.code(
    '''
import sympy as sp

a_sym = sp.Symbol('a')
mse_expr = 0
for x, y in zip(xs, ys):
    mse_expr += (a_sym * x - y)**2
mse_expr = mse_expr / len(xs)

dmse_da = sp.diff(mse_expr, a_sym)

# Convert to Python functions
mse_func = sp.lambdify(a_sym, mse_expr, modules='numpy')
dmse_da_func = sp.lambdify(a_sym, dmse_da, modules='numpy')
    ''',
    language="python"
)

st.markdown("**Now you can use `mse_func(a)` and `dmse_da_func(a)` in Python for any value of $a$.**")

# --- Gradient Descent for Best Fitting Line Through Origin using symbolic functions ---

a_val = 0.0  # initial guess
lr = 0.01   # learning rate
steps = 7

for step in range(steps):
    mse = mse_func(a_val)
    grad = dmse_da_func(a_val)
    print(f"Step {step+1}: a = {a_val:.4f}, MSE = {mse:.4f}")
    a_val = a_val - lr * grad

st.markdown("**Python code for gradient descent using the symbolic MSE and its derivative:**")
st.code(
    '''
a_val = 0.0  # initial guess
lr = 0.01   # learning rate
steps = 7

for step in range(steps):
    mse = mse_func(a_val)
    grad = dmse_da_func(a_val)
    print(f"Step {step+1}: a = {a_val:.4f}, MSE = {mse:.4f}")
    a_val = a_val - lr * grad
    ''',
    language="python"
)

# --- Gradient Descent for Best Fitting Line Through Origin using symbolic functions, with plots ---

steps = 7
a_val = 0.0  # initial guess
lr = 0.01    # learning rate

output_lines = []

# Fix y-axis limits for all plots
ymin = min(ys) - 5
ymax = max(ys) + 5

for step in range(steps):
    mse = mse_func(a_val)
    grad = dmse_da_func(a_val)
    output_lines.append(f"Step {step+1}: a = {a_val:.4f}, MSE = {mse:.4f}")

    # Plot the data and the current line (tiny size, no labels)
    fig, ax = plt.subplots(figsize=(1.2, 0.9), dpi=150)
    ax.scatter(xs, ys, color="red", s=8)
    ax.plot(xs, a_val * xs, color="blue", linewidth=1)
    ax.set_xlabel("x", fontsize=6)
    ax.set_ylabel("y", fontsize=6)
    ax.set_xlim(-1, 11)
    ax.set_ylim(ymin, ymax)
    ax.tick_params(axis='both', which='major', labelsize=5)
    legend = ax.get_legend()
    if legend is not None:
        legend.remove()

    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    col1, col2 = st.columns([1, 6])
    with col1:
        st.image(buf.getvalue(), use_container_width=True)
    with col2:
        st.markdown(f"`{output_lines[-1]}`")

    a_val = a_val - lr * grad  # <-- FIXED HERE

st.markdown("## Homework Problems: Downloadable Data Sets")

# Problem 1
np.random.seed(101)
xs1 = np.linspace(0, 9, 12)
ys1 = 1.5 * xs1 + np.random.normal(0, 3, size=12)
df1 = pd.DataFrame({'x': xs1, 'y': ys1})

st.markdown("### Homework 1")
st.download_button(
    label="Download Homework 1 Data (CSV)",
    data=df1.to_csv(index=False),
    file_name='homework1_points.csv',
    mime='text/csv'
)
st.markdown("""
**Instructions:**  
Use gradient descent to find the best fitting line through the origin (of the form $y = ax$) for the data.  
You may choose your own learning rate and number of steps.  
Plot the data and your best fitting line.  
What value of $a$ do you get after running gradient descent?  
How does it compare to the actual best fitting line through the origin?
""")
a1 = np.sum(xs1 * ys1) / np.sum(xs1**2)
st.markdown("**Solution:**  The best fitting line through the origin is:")
st.latex(fr"y = {a1:.3f}x")

# Problem 2
np.random.seed(202)
xs2 = np.linspace(-5, 5, 15)
ys2 = -2.0 * xs2 + np.random.normal(0, 2, size=15)
df2 = pd.DataFrame({'x': xs2, 'y': ys2})

st.markdown("### Homework 2")
st.download_button(
    label="Download Homework 2 Data (CSV)",
    data=df2.to_csv(index=False),
    file_name='homework2_points.csv',
    mime='text/csv'
)
st.markdown("""
**Instructions:**  
Use gradient descent to find the best fitting line through the origin (of the form $y = ax$) for the data.  
You may choose your own learning rate and number of steps.  
Plot the data and your best fitting line.  
What value of $a$ do you get after running gradient descent?  
How does it compare to the actual best fitting line through the origin?
""")
a2 = np.sum(xs2 * ys2) / np.sum(xs2**2)
st.markdown("**Solution:**  The best fitting line through the origin is:")
st.latex(fr"y = {a2:.3f}x")

# Problem 3
np.random.seed(303)
xs3 = np.linspace(2, 20, 18)
ys3 = 0.7 * xs3 + np.random.normal(0, 4, size=18)
df3 = pd.DataFrame({'x': xs3, 'y': ys3})

st.markdown("### Homework 3")
st.download_button(
    label="Download Homework 3 Data (CSV)",
    data=df3.to_csv(index=False),
    file_name='homework3_points.csv',
    mime='text/csv'
)
st.markdown("""
**Instructions:**  
Use gradient descent to find the best fitting line through the origin (of the form $y = ax$) for the data.  
You may choose your own learning rate and number of steps.  
Plot the data and your best fitting line.  
What value of $a$ do you get after running gradient descent?  
How does it compare to the actual best fitting line through the origin?
""")
a3 = np.sum(xs3 * ys3) / np.sum(xs3**2)
st.markdown("**Solution:**  The best fitting line through the origin is:")
st.latex(fr"y = {a3:.3f}x")