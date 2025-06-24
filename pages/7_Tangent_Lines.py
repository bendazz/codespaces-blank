import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

st.title("Interactive Tangent Line to $y = x^2$")

st.markdown(r"""
Move the slider to choose the $x$-coordinate of the point of tangency.  
The tangent line will be drawn at that point on the curve $y = x^2$.

""")

# Slider for the point of tangency
a = st.slider("Choose the x-coordinate of the point of tangency (a)", -5.0, 5.0, 1.0, step=0.1)

x = np.linspace(-5, 5, 400)
y = x**2

# Tangent line at x = a
tangent_y = 2*a*(x - a) + a**2

fig, ax = plt.subplots()
ax.plot(x, y, label=r"$y = x^2$")
ax.plot(x, tangent_y, '--', color='red', label=f"Tangent at x = {a:.1f}")
ax.scatter([a], [a**2], color='black', zorder=5, label=f"Point ({a:.1f}, {a**2:.2f})")
ax.axhline(0, color='gray', linewidth=0.5)
ax.axvline(0, color='gray', linewidth=0.5)
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_xlim(-5, 5)
ax.set_ylim(-5, 25)
ax.legend()
st.pyplot(fig)

st.header("Interactive Tangent Line to $y = x^3$")

st.markdown(r"""
Move the slider to choose the $x$-coordinate of the point of tangency.  
The tangent line will be drawn at that point on the curve $y = x^3$.
""")

a_cubic = st.slider("Choose the x-coordinate of the point of tangency for $y = x^3$ (a)", -3.0, 3.0, 1.0, step=0.1)

x_cubic = np.linspace(-3, 3, 400)
y_cubic = x_cubic**3

# Tangent line at x = a_cubic
tangent_y_cubic = 3*a_cubic**2*(x_cubic - a_cubic) + a_cubic**3

fig2, ax2 = plt.subplots()
ax2.plot(x_cubic, y_cubic, label=r"$y = x^3$")
ax2.plot(x_cubic, tangent_y_cubic, '--', color='red', label=f"Tangent at x = {a_cubic:.1f}")
ax2.scatter([a_cubic], [a_cubic**3], color='black', zorder=5, label=f"Point ({a_cubic:.1f}, {a_cubic**3:.2f})")
ax2.axhline(0, color='gray', linewidth=0.5)
ax2.axvline(0, color='gray', linewidth=0.5)
ax2.set_xlabel("x")
ax2.set_ylabel("y")
ax2.set_xlim(-3, 3)
ax2.set_ylim(-10, 10)
ax2.legend()
st.pyplot(fig2)

st.header("Interactive Tangent Line to $y = \sin(x)$")

st.markdown(r"""
Move the slider to choose the $x$-coordinate of the point of tangency.  
The tangent line will be drawn at that point on the curve $y = \sin(x)$.
""")

a_sin = st.slider("Choose the x-coordinate of the point of tangency for $y = \sin(x)$ (a)", -2 * np.pi, 2 * np.pi, 0.0, step=0.1)

x_sin = np.linspace(-2 * np.pi, 2 * np.pi, 400)
y_sin = np.sin(x_sin)

# Tangent line at x = a_sin
tangent_y_sin = np.cos(a_sin) * (x_sin - a_sin) + np.sin(a_sin)

fig3, ax3 = plt.subplots()
ax3.plot(x_sin, y_sin, label=r"$y = \sin(x)$")
ax3.plot(x_sin, tangent_y_sin, '--', color='red', label=f"Tangent at x = {a_sin:.2f}")
ax3.scatter([a_sin], [np.sin(a_sin)], color='black', zorder=5, label=f"Point ({a_sin:.2f}, {np.sin(a_sin):.2f})")
ax3.axhline(0, color='gray', linewidth=0.5)
ax3.axvline(0, color='gray', linewidth=0.5)
ax3.set_xlabel("x")
ax3.set_ylabel("y")
ax3.set_xlim(-2 * np.pi, 2 * np.pi)
ax3.set_ylim(-2, 2)
ax3.legend()
st.pyplot(fig3)