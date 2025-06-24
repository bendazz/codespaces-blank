import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

st.title("Step-by-Step Gradient Descent")

# ========== Example 1 ==========
st.subheader("Example 1")
st.latex(r"f(x) = x^2")

def f1(x):
    return x**2

def grad_f1(x):
    return 2*x

x0_1 = st.number_input("Enter starting $x$ (Example 1)", value=3.0, step=0.1, format="%.2f", key="x0_1")
lr_1 = st.number_input("Enter learning rate (α) (Example 1)", value=0.5, step=0.01, format="%.2f", key="lr_1")

if "x_history_1" not in st.session_state or st.session_state.get("last_x0_1") != x0_1 or st.session_state.get("last_lr_1") != lr_1:
    st.session_state.x_history_1 = [x0_1]
    st.session_state.last_x0_1 = x0_1
    st.session_state.last_lr_1 = lr_1

if st.button("Clear and Restart (Example 1)"):
    st.session_state.x_history_1 = [x0_1]
    st.session_state.last_x0_1 = x0_1
    st.session_state.last_lr_1 = lr_1

if st.button("Take a gradient descent step (Example 1)"):
    current_x = st.session_state.x_history_1[-1]
    next_x = current_x - lr_1 * grad_f1(current_x)
    st.session_state.x_history_1.append(next_x)

x_vals = np.linspace(-5, 5, 400)
y_vals = f1(x_vals)

fig, ax = plt.subplots()
ax.plot(x_vals, y_vals, label="$f(x) = x^2$")
ax.axhline(0, color='gray', linestyle=':', linewidth=1)
ax.axvline(0, color='gray', linestyle=':', linewidth=1)
ax.scatter([st.session_state.x_history_1[-1]], [0], color='blue', zorder=5, label="$x_{new}$")
ax.legend([r"$x_{new}$"], loc="upper center")
ax.set_xlabel("x")
ax.set_ylabel("f(x)")
ax.set_xlim(-5, 5)
ax.set_ylim(-1, max(25, f1(x0_1) + 2))
st.pyplot(fig)

# ========== Example 2 ==========
st.subheader("Example 2")
st.latex(r"f(x) = \sin(x)")

def f2(x):
    return np.sin(x)

def grad_f2(x):
    return np.cos(x)

x0_2 = st.number_input("Enter starting $x$ (Example 2)", value=2.0, step=0.1, format="%.2f", key="x0_2")
lr_2 = st.number_input("Enter learning rate (α) (Example 2)", value=0.5, step=0.01, format="%.2f", key="lr_2")

if "x_history_2" not in st.session_state or st.session_state.get("last_x0_2") != x0_2 or st.session_state.get("last_lr_2") != lr_2:
    st.session_state.x_history_2 = [x0_2]
    st.session_state.last_x0_2 = x0_2
    st.session_state.last_lr_2 = lr_2

if st.button("Clear and Restart (Example 2)"):
    st.session_state.x_history_2 = [x0_2]
    st.session_state.last_x0_2 = x0_2
    st.session_state.last_lr_2 = lr_2

if st.button("Take a gradient descent step (Example 2)"):
    current_x = st.session_state.x_history_2[-1]
    next_x = current_x - lr_2 * grad_f2(current_x)
    st.session_state.x_history_2.append(next_x)

x_vals2 = np.linspace(-2*np.pi, 2*np.pi, 400)
y_vals2 = f2(x_vals2)

fig2, ax2 = plt.subplots()
ax2.plot(x_vals2, y_vals2, label="$f(x) = \sin(x)$")
ax2.axhline(0, color='gray', linestyle=':', linewidth=1)
ax2.axvline(0, color='gray', linestyle=':', linewidth=1)
ax2.scatter([st.session_state.x_history_2[-1]], [0], color='blue', zorder=5, label="$x_{new}$")
ax2.legend([r"$x_{new}$"], loc="upper center")
ax2.set_xlabel("x")
ax2.set_ylabel("f(x)")
ax2.set_xlim(-2*np.pi, 2*np.pi)
ax2.set_ylim(-1.5, 1.5)
st.pyplot(fig2)

# ========== Example 3 ==========
st.subheader("Example 3")
st.latex(r"f(x) = e^x")

def f3(x):
    return np.exp(x)

def grad_f3(x):
    return np.exp(x)

x0_3 = st.number_input("Enter starting $x$ (Example 3)", value=1.0, step=0.1, format="%.2f", key="x0_3")
lr_3 = st.number_input("Enter learning rate (α) (Example 3)", value=0.2, step=0.01, format="%.2f", key="lr_3")

if "x_history_3" not in st.session_state or st.session_state.get("last_x0_3") != x0_3 or st.session_state.get("last_lr_3") != lr_3:
    st.session_state.x_history_3 = [x0_3]
    st.session_state.last_x0_3 = x0_3
    st.session_state.last_lr_3 = lr_3

if st.button("Clear and Restart (Example 3)"):
    st.session_state.x_history_3 = [x0_3]
    st.session_state.last_x0_3 = x0_3
    st.session_state.last_lr_3 = lr_3

if st.button("Take a gradient descent step (Example 3)"):
    current_x = st.session_state.x_history_3[-1]
    next_x = current_x - lr_3 * grad_f3(current_x)
    st.session_state.x_history_3.append(next_x)

x_vals3 = np.linspace(-2, 2, 400)
y_vals3 = f3(x_vals3)

fig3, ax3 = plt.subplots()
ax3.plot(x_vals3, y_vals3, label="$f(x) = e^x$")
ax3.axhline(0, color='gray', linestyle=':', linewidth=1)
ax3.axvline(0, color='gray', linestyle=':', linewidth=1)
ax3.scatter([st.session_state.x_history_3[-1]], [0], color='blue', zorder=5, label="$x_{new}$")
ax3.legend([r"$x_{new}$"], loc="upper center")
ax3.set_xlabel("x")
ax3.set_ylabel("f(x)")
ax3.set_xlim(-2, 2)
ax3.set_ylim(-1, np.exp(2) + 2)
st.pyplot(fig3)

# ========== Example 4 ==========
st.subheader("Example 4")
st.latex(r"f(x) = \sin(x) + 0.3x")

def f4(x):
    return np.sin(x) + 0.3 * x

def grad_f4(x):
    return np.cos(x) + 0.3

x0_4 = st.number_input("Enter starting $x$ (Example 4)", value=0.0, step=0.1, format="%.2f", key="x0_4")
lr_4 = st.number_input("Enter learning rate (α) (Example 4)", value=0.1, step=0.01, format="%.2f", key="lr_4")

if "x_history_4" not in st.session_state or st.session_state.get("last_x0_4") != x0_4 or st.session_state.get("last_lr_4") != lr_4:
    st.session_state.x_history_4 = [x0_4]
    st.session_state.last_x0_4 = x0_4
    st.session_state.last_lr_4 = lr_4

if st.button("Clear and Restart (Example 4)"):
    st.session_state.x_history_4 = [x0_4]
    st.session_state.last_x0_4 = x0_4
    st.session_state.last_lr_4 = lr_4

if st.button("Take a gradient descent step (Example 4)"):
    current_x = st.session_state.x_history_4[-1]
    next_x = current_x - lr_4 * grad_f4(current_x)
    st.session_state.x_history_4.append(next_x)

x_vals4 = np.linspace(-2, 16, 600)  # Wider domain to show more minima
y_vals4 = f4(x_vals4)

fig4, ax4 = plt.subplots()
ax4.plot(x_vals4, y_vals4, label=r"$f(x) = \sin(x) + 0.3x$")
ax4.axhline(0, color='gray', linestyle=':', linewidth=1)
ax4.axvline(0, color='gray', linestyle=':', linewidth=1)
ax4.scatter([st.session_state.x_history_4[-1]], [0], color='blue', zorder=5, label="$x_{new}$")
ax4.legend([r"$x_{new}$"], loc="upper center")
ax4.set_xlabel("x")
ax4.set_ylabel("f(x)")
ax4.set_xlim(-2, 16)
ax4.set_ylim(np.min(y_vals4)-1, np.max(y_vals4)+1)
st.pyplot(fig4)

