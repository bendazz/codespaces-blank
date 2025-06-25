import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

st.title("Horizontal Line Two Points Part 2: Gradient Descent")

# Two fixed points
point1 = (2, 3)
point2 = (7, 15)

# Numerical input for starting b and learning rate
start_b = st.number_input("Starting $b$", min_value=-5.0, max_value=20.0, value=4.00, step=0.01, format="%.2f")
lr = st.number_input("Learning rate (α)", min_value=0.001, max_value=2.0, value=0.1, step=0.001, format="%.3f")

# Session state for b history
if "b_history" not in st.session_state or st.session_state.get("b_start") != start_b:
    st.session_state.b_history = [start_b]
    st.session_state.b_start = start_b

# Gradient of MSE with respect to b
def grad_mse(b):
    return (2*(b - point1[1]) + 2*(b - point2[1])) / 2

# Button to take a gradient descent step
if st.button("Take a gradient descent step"):
    current_b = st.session_state.b_history[-1]
    new_b = current_b - lr * grad_mse(current_b)
    st.session_state.b_history.append(new_b)

# Current b is the last in history
b = st.session_state.b_history[-1]

# --- Two plots side by side ---
col1, col2 = st.columns(2)

with col1:
    fig, ax = plt.subplots()
    x_vals = np.linspace(0, 10, 100)
    y_vals = np.ones_like(x_vals) * b
    ax.plot(x_vals, y_vals, label=f"$y = {b:.2f}$", color="blue")
    ax.scatter([point1[0], point2[0]], [point1[1], point2[1]], color="red", zorder=5, label="Points")
    ax.scatter([point1[0], point2[0]], [b, b], color="blue", marker="o", zorder=6, label="Projection on $y=b$")
    ax.text(point1[0]+0.1, point1[1], f"({point1[0]}, {point1[1]})", color="red", va="bottom")
    ax.text(point2[0]+0.1, point2[1], f"({point2[0]}, {point2[1]})", color="red", va="bottom")
    ax.text(point1[0]+0.1, b, f"({point1[0]}, {b:.2f})", color="blue", va="bottom")
    ax.text(point2[0]+0.1, b, f"({point2[0]}, {b:.2f})", color="blue", va="bottom")
    ax.plot([point1[0], point1[0]], [point1[1], b], color="gray", linestyle="dotted")
    ax.plot([point2[0], point2[0]], [point2[1], b], color="gray", linestyle="dotted")
    diff1 = b - point1[1]
    diff2 = b - point2[1]
    ax.text(point1[0]+0.15, (point1[1]+b)/2, f"{diff1:+.2f}", color="gray", va="center", ha="left", fontsize=10, backgroundcolor="white")
    ax.text(point2[0]+0.15, (point2[1]+b)/2, f"{diff2:+.2f}", color="gray", va="center", ha="left", fontsize=10, backgroundcolor="white")
    ax.set_xlim(0, 10)
    ax.set_ylim(-5, 20)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend(loc="upper left")
    st.pyplot(fig)

with col2:
    # Plot MSE vs b for a range of b values
    b_vals = np.linspace(-5, 20, 400)
    mse_vals = [((b_test - point1[1])**2 + (b_test - point2[1])**2)/2 for b_test in b_vals]
    squared_error1 = (b - point1[1])**2
    squared_error2 = (b - point2[1])**2
    mse = (squared_error1 + squared_error2) / 2
    fig_mse, ax_mse = plt.subplots()
    ax_mse.plot(b_vals, mse_vals, color="purple")
    ax_mse.scatter([b], [mse], color="red", zorder=5, label="Current $(b, \mathrm{MSE})$")
    ax_mse.set_xlabel("b")
    ax_mse.set_ylabel("MSE")
    ax_mse.set_title("MSE vs b")
    ax_mse.legend()
    st.pyplot(fig_mse)



# --- LaTeX display of MSE and its derivative ---
st.markdown("**Mean Squared Error (MSE) as a function of $b$:**")
st.latex(
    r"\mathrm{MSE}(b) = \frac{1}{2}\left[(b - 3)^2 + (b - 15)^2\right]"
)

st.markdown("**Derivative of the MSE with respect to $b$:**")
st.latex(
    r"\frac{d\,\mathrm{MSE}}{db} = (b - 3) + (b - 15) = 2b - 18"
)

# --- Show how the new b is calculated via gradient descent ---
if len(st.session_state.b_history) > 1:
    old_b = st.session_state.b_history[-2]
    grad = grad_mse(old_b)
    st.markdown("**Gradient Descent Update Step:**")
    st.latex(
        rf"""
        \begin{{align*}}
        b_\text{{new}} &= b_\text{{old}} - \alpha \cdot \frac{{d\,\mathrm{{MSE}}}}{{db}} \bigg|_{{b = b_\text{{old}}}} \\
        &= {old_b:.4f} - {lr:.4f} \times ({grad:.4f}) \\
        &= {old_b - lr * grad:.4f}
        \end{{align*}}
        """
    )

st.markdown("---")
st.header("Homework: Gradient Descent Step for a Horizontal Line")

# Problem 1
st.markdown("### Problem 1")
st.markdown("Given points **(1, 4)** and **(5, 10)**, starting $b = 6$, and learning rate $\\alpha = 0.2$, what is the next value of $b$ after one gradient descent step?")
with st.expander("Solution"):
    st.latex(r"""
    \begin{align*}
    \text{Gradient:} \quad & (b - 4) + (b - 10) = 2b - 14 \\
    \text{At } b = 6: \quad & 2 \times 6 - 14 = 12 - 14 = -2 \\
    \text{Update:} \quad & b_{\text{new}} = 6 - 0.2 \times (-2) = 6 + 0.4 = 6.4
    \end{align*}
    """)

st.markdown("---")

# Problem 2
st.markdown("### Problem 2")
st.markdown("Given points **(2, -1)** and **(6, 3)**, starting $b = 0$, and learning rate $\\alpha = 0.1$, what is the next value of $b$ after one gradient descent step?")
with st.expander("Solution"):
    st.latex(r"""
    \begin{align*}
    \text{Gradient:} \quad & (b + 1) + (b - 3) = 2b - 2 \\
    \text{At } b = 0: \quad & 2 \times 0 - 2 = -2 \\
    \text{Update:} \quad & b_{\text{new}} = 0 - 0.1 \times (-2) = 0 + 0.2 = 0.2
    \end{align*}
    """)

st.markdown("---")

# Problem 3
st.markdown("### Problem 3")
st.markdown("Given points **(0, 0)** and **(10, 8)**, starting $b = 5$, and learning rate $\\alpha = 0.5$, what is the next value of $b$ after one gradient descent step?")
with st.expander("Solution"):
    st.latex(r"""
    \begin{align*}
    \text{Gradient:} \quad & (b - 0) + (b - 8) = 2b - 8 \\
    \text{At } b = 5: \quad & 2 \times 5 - 8 = 10 - 8 = 2 \\
    \text{Update:} \quad & b_{\text{new}} = 5 - 0.5 \times 2 = 5 - 1 = 4
    \end{align*}
    """)

st.markdown("---")

# Problem 4
st.markdown("### Problem 4")
st.markdown("Given points **(3, 7)** and **(9, 2)**, starting $b = 10$, and learning rate $\\alpha = 0.05$, what is the next value of $b$ after one gradient descent step?")
with st.expander("Solution"):
    st.latex(r"""
    \begin{align*}
    \text{Gradient:} \quad & (b - 7) + (b - 2) = 2b - 9 \\
    \text{At } b = 10: \quad & 2 \times 10 - 9 = 20 - 9 = 11 \\
    \text{Update:} \quad & b_{\text{new}} = 10 - 0.05 \times 11 = 10 - 0.55 = 9.45
    \end{align*}
    """)

st.markdown("---")

# Problem 5
st.markdown("### Problem 5")
st.markdown("Given points **(4, 12)** and **(8, 6)**, starting $b = 8$, and learning rate $\\alpha = 0.1$, what is the next value of $b$ after one gradient descent step?")
with st.expander("Solution"):
    st.latex(r"""
    \begin{align*}
    \text{Gradient:} \quad & (b - 12) + (b - 6) = 2b - 18 \\
    \text{At } b = 8: \quad & 2 \times 8 - 18 = 16 - 18 = -2 \\
    \text{Update:} \quad & b_{\text{new}} = 8 - 0.1 \times (-2) = 8 + 0.2 = 8.2
    \end{align*}
    """)