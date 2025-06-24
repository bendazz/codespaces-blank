import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

st.title("The General Equation of a Polynomial")

st.markdown(r"""
A **polynomial** is an expression consisting of variables (usually $x$) raised to non-negative integer powers, multiplied by coefficients, and summed together.

**General Equation (Degree $n$):**
$$
y = w_n x^n + w_{n-1} x^{n-1} + \cdots + w_1 x + w_0
$$

- $w_n, w_{n-1}, \ldots, w_1, w_0$ are coefficients.
- The highest power of $x$ (the degree) determines the general shape and number of turning points.

**Examples:**
- **Linear:** $y = 2x + 1$
- **Quadratic:** $y = x^2 - 3x + 2$
- **Cubic:** $y = -x^3 + 4x$
- **Quartic:** $y = x^4 - 2x^2 + 1$
""")

st.header("Examples of Polynomials")

examples = [
    {"coeffs": [1, 0], "desc": r"y = x"},
    {"coeffs": [1, -3, 2], "desc": r"y = x^2 - 3x + 2"},
    {"coeffs": [-1, 0, 4, 0], "desc": r"y = -x^3 + 4x"},
    {"coeffs": [1, 0, -2, 0, 1], "desc": r"y = x^4 - 2x^2 + 1"},
    {"coeffs": [0.5, -1, 0, 2], "desc": r"y = 0.5x^3 - x^2 + 2"},
]

x = np.linspace(-5, 5, 400)

for ex in examples:
    y = np.polyval(ex["coeffs"], x)
    st.latex(ex["desc"])
    fig, ax = plt.subplots()
    ax.plot(x, y, label=ex["desc"])
    ax.axhline(0, color='gray', linewidth=0.5)
    ax.axvline(0, color='gray', linewidth=0.5)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend()
    st.pyplot(fig)

st.header("Interactive Polynomial")

st.markdown(
    "Adjust the sliders to change the coefficients $w_4, w_3, w_2, w_1, w_0$ in the equation $y = w_4 x^4 + w_3 x^3 + w_2 x^2 + w_1 x + w_0$."
)

w4 = st.slider("w4 (x⁴ coefficient)", -2.0, 2.0, 0.0, step=0.1)
w3 = st.slider("w3 (x³ coefficient)", -2.0, 2.0, 0.0, step=0.1)
w2 = st.slider("w2 (x² coefficient)", -5.0, 5.0, 1.0, step=0.1)
w1 = st.slider("w1 (x coefficient)", -10.0, 10.0, 0.0, step=0.1)
w0 = st.slider("w0 (constant term)", -10.0, 10.0, 0.0, step=0.1)

coeffs = [w4, w3, w2, w1, w0]
y = np.polyval(coeffs, x)

st.latex(
    fr"y = {w4}x^4 + {w3}x^3 + {w2}x^2 + {w1}x + {w0}"
)

fig, ax = plt.subplots()
ax.plot(x, y, label=fr"$y = {w4}x^4 + {w3}x^3 + {w2}x^2 + {w1}x + {w0}$")
ax.axhline(0, color='gray', linewidth=0.5)
ax.axvline(0, color='gray', linewidth=0.5)
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_xlim(-5, 5)
ax.set_ylim(-20, 20)
ax.legend()
st.pyplot(fig)