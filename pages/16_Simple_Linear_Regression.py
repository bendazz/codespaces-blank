import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

st.title("Best Fitting Line Through Origin (Gradient Descent, 3 Points)")

# Three fixed points
point1 = (2, 3)
point2 = (7, 15)
point3 = (5, 8)

# Numerical input for starting a (slope) and learning rate
start_a = st.number_input(
    "Starting $a$ (slope)",
    min_value=-10.0,
    max_value=10.0,
    value=0.0,
    step=0.01,
    format="%.2f"
)
lr = st.number_input(
    "Learning rate (α)",
    min_value=0.001,
    max_value=2.0,
    value=0.05,
    step=0.001,
    format="%.3f"
)

# Session state for parameter history
if (
    "a_history" not in st.session_state
    or st.session_state.get("a_start") != start_a
):
    st.session_state.a_history = [start_a]
    st.session_state.a_start = start_a

# Clear button to reset the process
if st.button("Clear and Restart"):
    st.session_state.a_history = [start_a]
    st.session_state.a_start = start_a

# Gradient of MSE with respect to a
def grad_mse(a):
    xs = np.array([point1[0], point2[0], point3[0]])
    ys = np.array([point1[1], point2[1], point3[1]])
    preds = a * xs
    errors = preds - ys
    grad_a = (2 / 3) * np.sum(errors * xs)
    return grad_a

# Button to take a gradient descent step
if st.button("Take a gradient descent step"):
    current_a = st.session_state.a_history[-1]
    grad_a = grad_mse(current_a)
    new_a = current_a - lr * grad_a
    st.session_state.a_history.append(new_a)

# Current a is the last in history
a = st.session_state.a_history[-1]

# --- Two plots side by side ---
col1, col2 = st.columns(2)

with col1:
    fig, ax = plt.subplots()
    x_vals = np.linspace(0, 10, 100)
    y_vals = a * x_vals
    ax.plot(x_vals, y_vals, label=fr"$y = {a:.2f}x$", color="blue")
    # Red points
    xs = [point1[0], point2[0], point3[0]]
    ys = [point1[1], point2[1], point3[1]]
    ax.scatter(xs, ys, color="red", zorder=5, label="Points")
    # Blue projections (no labels)
    proj_ys = [a*x for x in xs]
    ax.scatter(xs, proj_ys, color="blue", marker="o", zorder=6, label="Projection on line")
    # Labels for red points only
    for x, y in zip(xs, ys):
        ax.text(x+0.1, y, f"({x}, {y})", color="red", va="bottom")
    ax.set_xlim(0, 10)
    ax.set_ylim(-5, 20)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend(loc="upper left")
    st.pyplot(fig)

with col2:
    # Plot MSE vs a for a range of a values
    a_vals = np.linspace(-10, 10, 400)
    mse_vals = []
    xs = np.array([point1[0], point2[0], point3[0]])
    ys = np.array([point1[1], point2[1], point3[1]])
    for a_test in a_vals:
        preds = a_test * xs
        mse = np.mean((preds - ys) ** 2)
        mse_vals.append(mse)
    # Current MSE
    mse = np.mean((a * xs - ys) ** 2)
    fig_mse, ax_mse = plt.subplots()
    ax_mse.plot(a_vals, mse_vals, color="purple")
    ax_mse.scatter([a], [mse], color="red", zorder=5, label="Current $(a, \mathrm{MSE})$")
    ax_mse.set_xlabel("a (slope)")
    ax_mse.set_ylabel("MSE")
    ax_mse.set_title("MSE vs a")
    ax_mse.set_xlim(-10, 10)
    ax_mse.set_ylim(-100, max(mse_vals)*1.1)
    ax_mse.legend()
    st.pyplot(fig_mse)



# --- Expanded MSE and derivative with actual values ---
st.markdown("**Expanded MSE and its derivative for these points:**")
st.latex(
    r"""
    \mathrm{MSE}(a) = \frac{1}{3}\left[(a \cdot 2 - 3)^2 + (a \cdot 7 - 15)^2 + (a \cdot 5 - 8)^2\right]
    """
)
st.latex(
    r"""
    \frac{d\,\mathrm{MSE}}{da} = \frac{2}{3}\left[(a \cdot 2 - 3) \cdot 2 + (a \cdot 7 - 15) \cdot 7 + (a \cdot 5 - 8) \cdot 5\right]
    """
)

st.markdown("---")
st.header("Homework")

# Problem 1
st.markdown("### Problem 1")
st.markdown("Given points **(1, 2)**, **(3, 7)**, **(5, 11)**, starting $a = 0$, and learning rate $\\alpha = 0.1$, what is the next value of $a$ after one gradient descent step?")
with st.expander("Solution"):
    st.latex(r"""
    \begin{align*}
    \mathrm{MSE}(a) &= \frac{1}{3}\left[(a \cdot 1 - 2)^2 + (a \cdot 3 - 7)^2 + (a \cdot 5 - 11)^2\right] \\
    \frac{d\,\mathrm{MSE}}{da} &= \frac{2}{3}\left[(a \cdot 1 - 2) \cdot 1 + (a \cdot 3 - 7) \cdot 3 + (a \cdot 5 - 11) \cdot 5\right] \\
    \text{At } a = 0: \\
    &= \frac{2}{3}\left[(-2) \cdot 1 + (-7) \cdot 3 + (-11) \cdot 5\right] \\
    &= \frac{2}{3}\left[-2 - 21 - 55\right] \\
    &= \frac{2}{3}(-78) = -52 \\
    a_{\text{new}} &= 0 - 0.1 \times (-52) = 0 + 5.2 = \boxed{5.2}
    \end{align*}
    """)

st.markdown("---")

# Problem 2
st.markdown("### Problem 2")
st.markdown("Given points **(2, 3)**, **(4, 8)**, **(6, 13)**, starting $a = 1$, and learning rate $\\alpha = 0.05$, what is the next value of $a$ after one gradient descent step?")
with st.expander("Solution"):
    st.latex(r"""
    \begin{align*}
    \mathrm{MSE}(a) &= \frac{1}{3}\left[(a \cdot 2 - 3)^2 + (a \cdot 4 - 8)^2 + (a \cdot 6 - 13)^2\right] \\
    \frac{d\,\mathrm{MSE}}{da} &= \frac{2}{3}\left[(a \cdot 2 - 3) \cdot 2 + (a \cdot 4 - 8) \cdot 4 + (a \cdot 6 - 13) \cdot 6\right] \\
    \text{At } a = 1: \\
    &= \frac{2}{3}\left[(-1) \cdot 2 + (-4) \cdot 4 + (-7) \cdot 6\right] \\
    &= \frac{2}{3}\left[-2 - 16 - 42\right] \\
    &= \frac{2}{3}(-60) = -40 \\
    a_{\text{new}} &= 1 - 0.05 \times (-40) = 1 + 2 = \boxed{3}
    \end{align*}
    """)

st.markdown("---")

# Problem 3
st.markdown("### Problem 3")
st.markdown("Given points **(1, 4)**, **(2, 7)**, **(3, 10)**, starting $a = 2$, and learning rate $\\alpha = 0.01$, what is the next value of $a$ after one gradient descent step?")
with st.expander("Solution"):
    st.latex(r"""
    \begin{align*}
    \mathrm{MSE}(a) &= \frac{1}{3}\left[(a \cdot 1 - 4)^2 + (a \cdot 2 - 7)^2 + (a \cdot 3 - 10)^2\right] \\
    \frac{d\,\mathrm{MSE}}{da} &= \frac{2}{3}\left[(a \cdot 1 - 4) \cdot 1 + (a \cdot 2 - 7) \cdot 2 + (a \cdot 3 - 10) \cdot 3\right] \\
    \text{At } a = 2: \\
    &= \frac{2}{3}\left[(2 - 4) \cdot 1 + (4 - 7) \cdot 2 + (6 - 10) \cdot 3\right] \\
    &= \frac{2}{3}\left[(-2) \cdot 1 + (-3) \cdot 2 + (-4) \cdot 3\right] \\
    &= \frac{2}{3}\left[-2 - 6 - 12\right] \\
    &= \frac{2}{3}(-20) = -13.33 \\
    a_{\text{new}} &= 2 - 0.01 \times (-13.33) = 2 + 0.1333 = \boxed{2.13}
    \end{align*}
    """)

st.markdown("---")

# Problem 4
st.markdown("### Problem 4")
st.markdown("Given points **(2, 5)**, **(3, 7)**, **(4, 9)**, starting $a = 0.5$, and learning rate $\\alpha = 0.2$, what is the next value of $a$ after one gradient descent step?")
with st.expander("Solution"):
    st.latex(r"""
    \begin{align*}
    \mathrm{MSE}(a) &= \frac{1}{3}\left[(a \cdot 2 - 5)^2 + (a \cdot 3 - 7)^2 + (a \cdot 4 - 9)^2\right] \\
    \frac{d\,\mathrm{MSE}}{da} &= \frac{2}{3}\left[(a \cdot 2 - 5) \cdot 2 + (a \cdot 3 - 7) \cdot 3 + (a \cdot 4 - 9) \cdot 4\right] \\
    \text{At } a = 0.5: \\
    &= \frac{2}{3}\left[(1 - 5) \cdot 2 + (1.5 - 7) \cdot 3 + (2 - 9) \cdot 4\right] \\
    &= \frac{2}{3}\left[(-4) \cdot 2 + (-5.5) \cdot 3 + (-7) \cdot 4\right] \\
    &= \frac{2}{3}\left[-8 - 16.5 - 28\right] \\
    &= \frac{2}{3}(-52.5) = -35 \\
    a_{\text{new}} &= 0.5 - 0.2 \times (-35) = 0.5 + 7 = \boxed{7.5}
    \end{align*}
    """)

st.markdown("---")

# Problem 5
st.markdown("### Problem 5")
st.markdown("Given points **(1, 1)**, **(2, 2)**, **(3, 3)**, starting $a = 0$, and learning rate $\\alpha = 0.5$, what is the next value of $a$ after one gradient descent step?")
with st.expander("Solution"):
    st.latex(r"""
    \begin{align*}
    \mathrm{MSE}(a) &= \frac{1}{3}\left[(a \cdot 1 - 1)^2 + (a \cdot 2 - 2)^2 + (a \cdot 3 - 3)^2\right] \\
    \frac{d\,\mathrm{MSE}}{da} &= \frac{2}{3}\left[(a \cdot 1 - 1) \cdot 1 + (a \cdot 2 - 2) \cdot 2 + (a \cdot 3 - 3) \cdot 3\right] \\
    \text{At } a = 0: \\
    &= \frac{2}{3}\left[(-1) \cdot 1 + (-2) \cdot 2 + (-3) \cdot 3\right] \\
    &= \frac{2}{3}\left[-1 - 4 - 9\right] \\
    &= \frac{2}{3}(-14) = -9.33 \\
    a_{\text{new}} &= 0 - 0.5 \times (-9.33) = 0 + 4.67 = \boxed{4.67}
    \end{align*}
    """)