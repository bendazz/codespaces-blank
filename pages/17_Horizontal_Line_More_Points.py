import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import io

st.title("Simple Linear Regression with More Points (Horizontal Line)")

# Generate 10 random points in the plane (for reproducibility, set a seed)
np.random.seed(42)
xs = np.linspace(0, 10, 10)
ys = 2 * xs + 1 + np.random.normal(0, 3, size=10)  # Some linear trend + noise

# Fixed value for b
b = 5.0

fig, ax = plt.subplots()
ax.scatter(xs, ys, color="red", label="Data points")
ax.plot([xs[0], xs[-1]], [b, b], color="blue", label=fr"$y = {b:.1f}$")
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

import sympy as sp

# Symbolic variable for b
b_sym = sp.Symbol('b')

# Calculate MSE symbolically, point by point
mse_expr = 0
for x, y in zip(xs, ys):
    mse_expr += (b_sym - y)**2
mse_expr = mse_expr / len(xs)

# Derivative of MSE with respect to b
dmse_db = sp.diff(mse_expr, b_sym)

# Convert symbolic expressions to Python functions
mse_func = sp.lambdify(b_sym, mse_expr, modules='numpy')
dmse_db_func = sp.lambdify(b_sym, dmse_db, modules='numpy')

st.markdown("**Python code to create the symbolic MSE, its derivative, and convert them to Python functions:**")
st.code(
    '''
import sympy as sp

b_sym = sp.Symbol('b')
mse_expr = 0
for x, y in zip(xs, ys):
    mse_expr += (b_sym - y)**2
mse_expr = mse_expr / len(xs)

dmse_db = sp.diff(mse_expr, b_sym)

# Convert to Python functions
mse_func = sp.lambdify(b_sym, mse_expr, modules='numpy')
dmse_db_func = sp.lambdify(b_sym, dmse_db, modules='numpy')
    ''',
    language="python"
)

st.markdown("**Now you can use `mse_func(b)` and `dmse_db_func(b)` in Python for any value of $b$.**")


# --- Gradient Descent for Best Fitting Horizontal Line ---





# --- Gradient Descent for Best Fitting Horizontal Line using symbolic functions ---

b_val = 0.0  # initial guess
lr = 0.05    # learning rate
steps = 10

b_vals = []
mse_vals = []

for step in range(steps):
    mse = mse_func(b_val)
    grad = dmse_db_func(b_val)
    b_vals.append(b_val)
    mse_vals.append(mse)
    print(f"Step {step+1}: b = {b_val:.4f}, MSE = {mse:.4f}")
    b_val = b_val - lr * grad

# Display the code for students
st.markdown("**Python code for gradient descent using the symbolic MSE and its derivative:**")
st.code(
    '''
b_val = 0.0  # initial guess
lr = 0.05    # learning rate
steps = 30

for step in range(steps):
    mse = mse_func(b_val)
    grad = dmse_db_func(b_val)
    print(f"Step {step+1}: b = {b_val:.4f}, MSE = {mse:.4f}")
    b_val = b_val - lr * grad
    ''',
    language="python"
)

# Display what the output looks like, with a much smaller plot for each step and no point labels
steps = 30  # number of gradient descent steps

output_lines = []
b_val = 0.0
for step in range(steps):
    mse = mse_func(b_val)
    grad = dmse_db_func(b_val)
    output_lines.append(f"Step {step+1}: b = {b_val:.4f}, MSE = {mse:.4f}")

    # Plot the data and the current horizontal line (tiny size, no labels)
    fig, ax = plt.subplots(figsize=(1.2, 0.9), dpi=150)
    ax.scatter(xs, ys, color="red", s=8)
    ax.plot([xs[0], xs[-1]], [b_val, b_val], color="blue", linewidth=1)
    ax.set_xlabel("x", fontsize=6)
    ax.set_ylabel("y", fontsize=6)
    ax.set_xlim(-1, 11)
    ax.set_ylim(min(ys)-5, max(ys)+5)
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

    b_val = b_val - lr * grad

# --- Homework Problems: Downloadable Data Sets for Students ---

import pandas as pd
import numpy as np

st.markdown("## Homework Problems: Downloadable Data Sets")

# Problem 1
np.random.seed(101)
xs1 = np.linspace(0, 9, 12)
ys1 = 3 * xs1 + 2 + np.random.normal(0, 4, size=12)
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
Use gradient descent to find the best fitting horizontal line (of the form $y = b$) for the data.  
You may choose your own learning rate and number of steps.  
Plot the data and your best fitting line.  
What value of $b$ do you get after running gradient descent?  
How does it compare to the actual best fitting horizontal line?
""")
b1 = ys1.mean()
st.markdown("**Solution:**  The best fitting horizontal line is at the mean of the $y$-values:")
st.latex(fr"y = {b1:.2f}")

# Problem 2
np.random.seed(202)
xs2 = np.linspace(-5, 5, 15)
ys2 = -1.5 * xs2 + 7 + np.random.normal(0, 2, size=15)
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
Use gradient descent to find the best fitting horizontal line (of the form $y = b$) for the data.  
You may choose your own learning rate and number of steps.  
Plot the data and your best fitting line.  
What value of $b$ do you get after running gradient descent?  
How does it compare to the actual best fitting horizontal line?
""")
b2 = ys2.mean()
st.markdown("**Solution:**  The best fitting horizontal line is at the mean of the $y$-values:")
st.latex(fr"y = {b2:.2f}")

# Problem 3
np.random.seed(303)
xs3 = np.linspace(2, 20, 18)
ys3 = 0.5 * xs3 - 4 + np.random.normal(0, 3, size=18)
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
Use gradient descent to find the best fitting horizontal line (of the form $y = b$) for the data.  
You may choose your own learning rate and number of steps.  
Plot the data and your best fitting line.  
What value of $b$ do you get after running gradient descent?  
How does it compare to the actual best fitting horizontal line?
""")
b3 = ys3.mean()
st.markdown("**Solution:**  The best fitting horizontal line is at the mean of the $y$-values:")
st.latex(fr"y = {b3:.2f}")
