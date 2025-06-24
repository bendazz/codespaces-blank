import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

st.title("Horizontal Line Two Points Part 1")

# Two fixed points
point1 = (2, 3)
point2 = (7, 15)

# Slider for large adjustments
b_slider = st.slider("Choose $b$ (slider, coarse)", min_value=-5.0, max_value=20.0, value=4.00, step=0.1)

# Number input for fine adjustments, initialized to slider value
b = st.number_input("Fine-tune $b$ (number input, fine)", min_value=-5.0, max_value=20.0, value=float(b_slider), step=0.01, format="%.2f")

# Plot
fig, ax = plt.subplots()
x_vals = np.linspace(0, 10, 100)
y_vals = np.ones_like(x_vals) * b
ax.plot(x_vals, y_vals, label=f"$y = {b}$", color="blue")
ax.scatter([point1[0], point2[0]], [point1[1], point2[1]], color="red", zorder=5, label="Points")

# Points on the horizontal line aligned with the red points
ax.scatter([point1[0], point2[0]], [b, b], color="blue", marker="o", zorder=6, label="Projection on $y=b$")

# Label each red point with its coordinates
ax.text(point1[0]+0.1, point1[1], f"({point1[0]}, {point1[1]})", color="red", va="bottom")
ax.text(point2[0]+0.1, point2[1], f"({point2[0]}, {point2[1]})", color="red", va="bottom")

# Label each blue point (projection) with its coordinates
ax.text(point1[0]+0.1, b, f"({point1[0]}, {b:.2f})", color="blue", va="bottom")
ax.text(point2[0]+0.1, b, f"({point2[0]}, {b:.2f})", color="blue", va="bottom")

# Dotted lines from each point to the horizontal line
ax.plot([point1[0], point1[0]], [point1[1], b], color="gray", linestyle="dotted")
ax.plot([point2[0], point2[0]], [point2[1], b], color="gray", linestyle="dotted")

# Label the dotted lines with their signed length
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

# Calculate and display mean squared error with explanation
squared_error1 = (b - point1[1])**2
squared_error2 = (b - point2[1])**2
mse = (squared_error1 + squared_error2) / 2

st.markdown("**Mean Squared Error (MSE) Calculation:**")

st.latex(r"\text{MSE} = \frac{(b - y_1)^2 + (b - y_2)^2}{2}")

st.latex(
    rf"\text{{MSE}} = \frac{{({b:.2f} - {point1[1]})^2 + ({b:.2f} - {point2[1]})^2}}{{2}}"
)

st.latex(
    rf"\text{{MSE}} = \frac{{({b - point1[1]:.2f})^2 + ({b - point2[1]:.2f})^2}}{{2}}"
)

st.latex(
    rf"\text{{MSE}} = \frac{{{squared_error1:.2f} + {squared_error2:.2f}}}{{2}}"
)

st.latex(
    rf"\text{{MSE}} = {mse:.2f}"
)

st.markdown("---")
st.header("Homework Problems: Mean Squared Error for a Horizontal Line")

# Problem 1
st.markdown("### Problem 1")
st.markdown("Given points **A(1, 5)** and **B(4, 9)**, and the horizontal line $y = 7$, calculate the mean squared error (MSE).")
with st.expander("Solution"):
    st.latex(r"""
    \begin{align*}
    \text{MSE} &= \frac{(7 - 5)^2 + (7 - 9)^2}{2} \\
               &= \frac{(2)^2 + (-2)^2}{2} \\
               &= \frac{4 + 4}{2} \\
               &= \frac{8}{2} = 4
    \end{align*}
    """)

st.markdown("---")

# Problem 2
st.markdown("### Problem 2")
st.markdown("Given points **A(2, -3)** and **B(6, 1)**, and the horizontal line $y = 0$, calculate the mean squared error (MSE).")
with st.expander("Solution"):
    st.latex(r"""
    \begin{align*}
    \text{MSE} &= \frac{(0 - (-3))^2 + (0 - 1)^2}{2} \\
               &= \frac{(3)^2 + (-1)^2}{2} \\
               &= \frac{9 + 1}{2} \\
               &= \frac{10}{2} = 5
    \end{align*}
    """)

st.markdown("---")

# Problem 3
st.markdown("### Problem 3")
st.markdown("Given points **A(0, 0)** and **B(10, 8)**, and the horizontal line $y = 3$, calculate the mean squared error (MSE).")
with st.expander("Solution"):
    st.latex(r"""
    \begin{align*}
    \text{MSE} &= \frac{(3 - 0)^2 + (3 - 8)^2}{2} \\
               &= \frac{(3)^2 + (-5)^2}{2} \\
               &= \frac{9 + 25}{2} \\
               &= \frac{34}{2} = 17
    \end{align*}
    """)

st.markdown("---")

# Problem 4
st.markdown("### Problem 4")
st.markdown("Given points **A(3, 7)** and **B(9, 2)**, and the horizontal line $y = 5$, calculate the mean squared error (MSE).")
with st.expander("Solution"):
    st.latex(r"""
    \begin{align*}
    \text{MSE} &= \frac{(5 - 7)^2 + (5 - 2)^2}{2} \\
               &= \frac{(-2)^2 + (3)^2}{2} \\
               &= \frac{4 + 9}{2} \\
               &= \frac{13}{2} = 6.5
    \end{align*}
    """)

st.markdown("---")

# Problem 5
st.markdown("### Problem 5")
st.markdown("Given points **A(4, 12)** and **B(8, 6)**, and the horizontal line $y = 10$, calculate the mean squared error (MSE).")
with st.expander("Solution"):
    st.latex(r"""
    \begin{align*}
    \text{MSE} &= \frac{(10 - 12)^2 + (10 - 6)^2}{2} \\
               &= \frac{(-2)^2 + (4)^2}{2} \\
               &= \frac{4 + 16}{2} \\
               &= \frac{20}{2} = 10
    \end{align*}
    """)