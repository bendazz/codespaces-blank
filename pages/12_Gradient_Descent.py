import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrow

st.title("Gradient Descent Visualization")

def f(x):
    return x**2

def grad_f(x):
    return 2*x

# User input for starting x and learning rate
x0 = st.number_input("Enter starting $x$", value=3.0, step=0.1, format="%.2f")
lr = st.number_input("Enter learning rate (α)", value=0.5, step=0.01, format="%.2f")

# Compute negative gradient step
g0 = grad_f(x0)
step = -lr * g0
x1 = x0 + step

# Plot setup
x_vals = np.linspace(-5, 5, 400)
y_vals = f(x_vals)

fig, ax = plt.subplots()
ax.plot(x_vals, y_vals, label="$f(x) = x^2$")
ax.axhline(0, color='gray', linestyle=':', linewidth=1)
ax.axvline(0, color='gray', linestyle=':', linewidth=1)

# Draw starting point on x-axis
pt0 = ax.scatter([x0], [0], color='red', zorder=5, label=fr"$x_0 = {x0:.2f}$")

# Draw negative gradient vector (scaled by learning rate) on x-axis
arrow = ax.arrow(
    x0, 0, step, 0,
    head_width=0.8, head_length=0.3, fc='green', ec='green',
    length_includes_head=True
)

# Draw next point at tip of vector
pt1 = ax.scatter([x1], [0], color='blue', zorder=5, label=fr"$x_1 = {x1:.2f}$")

# Custom legend entry for the step, using a FancyArrow
arrow_legend = FancyArrow(0, 0, 1, 0, width=0.01, length_includes_head=True, color='green')
custom_lines = [
    Line2D([0], [0], color='red', marker='o', linestyle='', label=fr"$x_0 = {x0:.2f}$"),
    Line2D([0], [0], color='blue', marker='o', linestyle='', label=fr"$x_1 = {x1:.2f}$"),
    arrow_legend,
    Line2D([0], [0], color='black', linestyle='-', label="$f(x) = x^2$")
]
custom_labels = [
    fr"$x_0 = {x0:.2f}$",
    fr"$x_1 = {x1:.2f}$",
    r"Step: $x_1 = x_0 - \alpha f'(x_0)$",
    "$f(x) = x^2$"
]
ax.legend(custom_lines, custom_labels)
ax.set_xlabel("x")
ax.set_ylabel("f(x)")
ax.set_xlim(-5, 5)
ax.set_ylim(-1, max(25, f(x0) + 2))
st.pyplot(fig)

# Show step details
st.latex(rf"f(x) = x^2")
st.latex(rf"f'(x) = 2x")
st.latex(rf"f'({x0}) = 2 \times {x0} = {g0}")
st.latex(rf"\text{{Step:}}\quad x_1 = x_0 - \alpha f'(x_0) = {x0} - {lr} \times {g0} = {x1}")