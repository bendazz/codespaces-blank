import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

st.title("The General Equation of a Parabola")

st.markdown(r"""
A **parabola** is a U-shaped curve that can open upwards, downwards, left, or right.  
The most common form is the **vertical parabola**, which opens up or down.

**General Equation (Standard Form):**

$$
y = w_2 x^2 + w_1 x + w_0
$$

- **w₂** is the quadratic coefficient (controls the "width" and direction).
- **w₁** is the linear coefficient (controls the tilt).
- **w₀** is the constant term (controls the vertical shift).
- If **w₂ > 0**, the parabola opens upwards.
- If **w₂ < 0**, the parabola opens downwards.
""")

st.header("Examples of Parabolas")

examples = [
    {"w2": 1, "w1": 0, "w0": 0, "desc": r"y = x^2"},
    {"w2": -1, "w1": 0, "w0": 0, "desc": r"y = -x^2"},
    {"w2": 1, "w1": 2, "w0": 1, "desc": r"y = x^2 + 2x + 1"},
    {"w2": 0.5, "w1": 0, "w0": -2, "desc": r"y = 0.5x^2 - 2"},
    {"w2": 2, "w1": -4, "w0": 1, "desc": r"y = 2x^2 - 4x + 1"},
    {"w2": -0.5, "w1": 3, "w0": -1, "desc": r"y = -0.5x^2 + 3x - 1"},
    {"w2": 1, "w1": 0, "w0": -9, "desc": r"y = x^2 - 9"},
]

x = np.linspace(-10, 10, 400)

for ex in examples:
    y = ex["w2"] * x**2 + ex["w1"] * x + ex["w0"]
    st.latex(ex["desc"])
    fig, ax = plt.subplots()
    ax.plot(x, y, label=ex["desc"])
    ax.axhline(0, color='gray', linewidth=0.5)
    ax.axvline(0, color='gray', linewidth=0.5)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend()
    st.pyplot(fig)

st.header("Interactive Parabola")

st.markdown("Adjust the sliders to change the values of $w_2$, $w_1$, and $w_0$ in the equation $y = w_2 x^2 + w_1 x + w_0$.")

w2 = st.slider("w2 (quadratic coefficient)", -5.0, 5.0, 1.0, step=0.1)
w1 = st.slider("w1 (linear coefficient)", -10.0, 10.0, 0.0, step=0.1)
w0 = st.slider("w0 (constant term)", -20.0, 20.0, 0.0, step=0.1)

x = np.linspace(-10, 10, 400)
y = w2 * x**2 + w1 * x + w0

st.latex(fr"y = {w2}x^2 + {w1}x + {w0}")

fig, ax = plt.subplots()
ax.plot(x, y, label=fr"$y = {w2}x^2 + {w1}x + {w0}$")
ax.axhline(0, color='gray', linewidth=0.5)
ax.axvline(0, color='gray', linewidth=0.5)
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_xlim(-10, 10)
ax.set_ylim(-20, 20)
ax.legend()
st.pyplot(fig)

