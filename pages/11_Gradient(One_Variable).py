import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

st.title("Gradient Visualization ")

# ========== Example 1 ==========
st.subheader("Example 1: $f(x) = x^2$")

def f1(x):
    return x**2

def grad_f1(x):
    return 2*x

x0 = st.slider("Choose a value for $x$ (Example 1)", -5.0, 5.0, 2.0, step=0.1)
y0 = f1(x0)
g0 = grad_f1(x0)

st.latex(r"f(x) = x^2")
st.latex(r"f'(x) = 2x")
st.latex(rf"f'({x0}) = 2 \times {x0} = {g0}")

x_vals = np.linspace(-5, 5, 400)
y_vals = f1(x_vals)

# For Example 1
fig, ax = plt.subplots()
ax.plot(x_vals, y_vals, label=r"$f(x) = x^2$")
# Place the red point on the x-axis at (x0, 0)
ax.scatter([x0], [0], color='red', zorder=5, label=fr"Point $({x0:.2f}, 0)$")
arrow_dx = g0
arrow_dy = 0
ax.arrow(x0, 0, arrow_dx, arrow_dy, head_width=0.8, head_length=0.3, fc='green', ec='green', length_includes_head=True, label="Gradient Vector (on x-axis)")
ax.axhline(0, color='gray', linestyle=':', linewidth=1)
ax.axvline(0, color='gray', linestyle=':', linewidth=1)
ax.set_xlabel("x")
ax.set_ylabel("f(x)")
ax.set_xlim(-5, 5)
ax.set_ylim(-1, max(25, y0 + 2))
ax.legend()
st.pyplot(fig)

# ========== Example 2 ==========
st.subheader("Example 2: $f(x) = \sin(x)$")

def f2(x):
    return np.sin(x)

def grad_f2(x):
    return np.cos(x)

x1 = st.slider("Choose a value for $x$ (Example 2)", -2*np.pi, 2*np.pi, 1.0, step=0.1)
y1 = f2(x1)
g1 = grad_f2(x1)

st.latex(r"f(x) = \sin(x)")
st.latex(r"f'(x) = \cos(x)")
st.latex(rf"f'({x1}) = \cos({x1:.2f}) = {g1:.4f}")

x_vals2 = np.linspace(-2*np.pi, 2*np.pi, 400)
y_vals2 = f2(x_vals2)

# For Example 2
fig2, ax2 = plt.subplots()
ax2.plot(x_vals2, y_vals2, label=r"$f(x) = \sin(x)$")
ax2.scatter([x1], [0], color='red', zorder=5, label=fr"Point $({x1:.2f}, 0)$")
arrow_dx2 = g1
arrow_dy2 = 0
ax2.arrow(x1, 0, arrow_dx2, arrow_dy2, head_width=0.2, head_length=0.1, fc='green', ec='green', length_includes_head=True, label="Gradient Vector (on x-axis)")
ax2.axhline(0, color='gray', linestyle=':', linewidth=1)
ax2.axvline(0, color='gray', linestyle=':', linewidth=1)
ax2.set_xlabel("x")
ax2.set_ylabel("f(x)")
ax2.set_xlim(-2*np.pi, 2*np.pi)
ax2.set_ylim(-1.5, 1.5)
ax2.legend()
st.pyplot(fig2)

# ========== Example 3 ==========
st.subheader("Example 3: $f(x) = e^x$")

def f3(x):
    return np.exp(x)

def grad_f3(x):
    return np.exp(x)

x2 = st.slider("Choose a value for $x$ (Example 3)", -2.0, 2.0, 0.5, step=0.1)
y2 = f3(x2)
g2 = grad_f3(x2)

st.latex(r"f(x) = e^x")
st.latex(r"f'(x) = e^x")
st.latex(rf"f'({x2}) = e^{{{x2:.2f}}} = {g2:.4f}")

x_vals3 = np.linspace(-2, 2, 400)
y_vals3 = f3(x_vals3)

# For Example 3
fig3, ax3 = plt.subplots()
ax3.plot(x_vals3, y_vals3, label=r"$f(x) = e^x$")
ax3.scatter([x2], [0], color='red', zorder=5, label=fr"Point $({x2:.2f}, 0)$")
arrow_dx3 = g2
arrow_dy3 = 0
ax3.arrow(x2, 0, arrow_dx3, arrow_dy3, head_width=0.2, head_length=0.1, fc='green', ec='green', length_includes_head=True, label="Gradient Vector (on x-axis)")
ax3.axhline(0, color='gray', linestyle=':', linewidth=1)
ax3.axvline(0, color='gray', linestyle=':', linewidth=1)
ax3.set_xlabel("x")
ax3.set_ylabel("f(x)")
ax3.set_xlim(-2, 2)
ax3.set_ylim(-0.5, np.exp(2)+1)
ax3.legend()
st.pyplot(fig3)