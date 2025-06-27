import streamlit as st
import numpy as np
import plotly.graph_objs as go


st.header("Gradient 3D")

st.subheader("Example 1")

st.markdown(
    r"""
    Let's explore the **gradient** of the surface $z = x^2 + y^2$.
    - Select a point $(x_0, y_0)$ on the surface.
    - The tangent lines to the $x$-trace (holding $y$ fixed) and $y$-trace (holding $x$ fixed) at that point will be shown.
    - These lines represent the slopes in the $x$ and $y$ directions, which together form the gradient vector.
    """
)

# Surface grid
x = np.linspace(-10, 10, 60)
y = np.linspace(-10, 10, 60)
X, Y = np.meshgrid(x, y)
Z = X**2 + Y**2

# Student selects a point (x0, y0)
colx, coly = st.columns(2)
with colx:
    x0 = st.slider("Select $x_0$", float(x[0]), float(x[-1]), 2.0, 0.1, key="grad_x0")
with coly:
    y0 = st.slider("Select $y_0$", float(y[0]), float(y[-1]), 3.0, 0.1, key="grad_y0")
z0 = x0**2 + y0**2

# x-trace: y = y0, vary x
x_trace = np.linspace(-10, 10, 200)
y_trace_x = np.full_like(x_trace, y0)
z_trace_x = x_trace**2 + y0**2

# y-trace: x = x0, vary y
y_trace = np.linspace(-10, 10, 200)
x_trace_y = np.full_like(y_trace, x0)
z_trace_y = x0**2 + y_trace**2

# Tangent to x-trace at (x0, y0)
dz_dx = 2 * x0
x_tan = np.linspace(x0 - 5, x0 + 5, 100)
z_tan_x = z0 + dz_dx * (x_tan - x0)
y_tan_x = np.full_like(x_tan, y0)

# Tangent to y-trace at (x0, y0)
dz_dy = 2 * y0
y_tan = np.linspace(y0 - 5, y0 + 5, 100)
z_tan_y = z0 + dz_dy * (y_tan - y0)
x_tan_y = np.full_like(y_tan, x0)

# 3D plot
surface = go.Surface(x=x, y=y, z=Z, colorscale="Viridis", opacity=0.7, showscale=False)
trace_x = go.Scatter3d(
    x=x_trace, y=y_trace_x, z=z_trace_x,
    mode="lines", line=dict(color="red", width=5), name="x-trace"
)
trace_y = go.Scatter3d(
    x=x_trace_y, y=y_trace, z=z_trace_y,
    mode="lines", line=dict(color="blue", width=5), name="y-trace"
)
tan_x = go.Scatter3d(
    x=x_tan, y=y_tan_x, z=z_tan_x,
    mode="lines", line=dict(color="orange", width=6), name="Tangent to x-trace"
)
tan_y = go.Scatter3d(
    x=x_tan_y, y=y_tan, z=z_tan_y,
    mode="lines", line=dict(color="green", width=6), name="Tangent to y-trace"
)
pt = go.Scatter3d(
    x=[x0], y=[y0], z=[z0],
    mode="markers", marker=dict(size=8, color="black"), name="Selected point"
)
axes = [
    go.Scatter3d(x=[-10, 10], y=[0, 0], z=[0, 0], mode='lines', line=dict(color='black', width=4)),
    go.Scatter3d(x=[0, 0], y=[-10, 10], z=[0, 0], mode='lines', line=dict(color='black', width=4)),
    go.Scatter3d(x=[0, 0], y=[0, 0], z=[0, 200], mode='lines', line=dict(color='black', width=4))
]

# Add a point in the xy-plane (projection of the selected point)
pt_xy = go.Scatter3d(
    x=[x0], y=[y0], z=[0],
    mode="markers",
    marker=dict(size=8, color="purple", symbol="circle"),
    name="Projection in xy-plane"
)

# Draw the gradient vector at (x0, y0, 0) in the xy-plane
# Gradient is (2x0, 2y0, 0), but for visualization, scale it for clarity
grad_scale = 2.5  # Adjust for visual clarity
grad_x = 2 * x0
grad_y = 2 * y0
arrow = go.Cone(
    x=[x0], y=[y0], z=[0],
    u=[grad_scale * grad_x], v=[grad_scale * grad_y], w=[0],
    sizemode="absolute", sizeref=10,
    anchor="tail",
    showscale=False,
    colorscale=[[0, "magenta"], [1, "magenta"]],
    name="Gradient vector"
)

# 2D gradient vectors at (x0, y0, 0)
arrow_scale = 1.2  # Adjust for visual clarity

# X component vector: (2x0, 0)
xvec_end = (x0 + arrow_scale * grad_x, y0, 0)
xvec = go.Scatter3d(
    x=[x0, xvec_end[0]],
    y=[y0, xvec_end[1]],
    z=[0, 0],
    mode="lines+markers",
    line=dict(color="red", width=7),
    marker=dict(size=[4, 10], color="red"),
    name="Gradient x-component",
    showlegend=False
)

# Y component vector: (0, 2y0)
yvec_end = (x0, y0 + arrow_scale * grad_y, 0)
yvec = go.Scatter3d(
    x=[x0, yvec_end[0]],
    y=[y0, yvec_end[1]],
    z=[0, 0],
    mode="lines+markers",
    line=dict(color="blue", width=7),
    marker=dict(size=[4, 10], color="blue"),
    name="Gradient y-component",
    showlegend=False
)

# Full gradient vector: (2x0, 2y0)
grad_end = (x0 + arrow_scale * grad_x, y0 + arrow_scale * grad_y, 0)
gradvec = go.Scatter3d(
    x=[x0, grad_end[0]],
    y=[y0, grad_end[1]],
    z=[0, 0],
    mode="lines+markers",
    line=dict(color="magenta", width=7),
    marker=dict(size=[4, 10], color="magenta"),
    name="Gradient vector",
    showlegend=False
)

# Add a dotted line from the surface point to its projection in the xy-plane
dotted_line = go.Scatter3d(
    x=[x0, x0],
    y=[y0, y0],
    z=[z0, 0],
    mode="lines",
    line=dict(color="purple", width=4, dash="dot"),
    showlegend=False,
    name="Projection line"
)

# Dotted line parallel to x-axis at y = y0 in the z=0 plane
dotted_x = go.Scatter3d(
    x=[x[0], x[-1]],
    y=[y0, y0],
    z=[0, 0],
    mode="lines",
    line=dict(color="gray", width=3, dash="dot"),
    showlegend=False,
    name="x-axis trace"
)

# Dotted line parallel to y-axis at x = x0 in the z=0 plane
dotted_y = go.Scatter3d(
    x=[x0, x0],
    y=[y[0], y[-1]],
    z=[0, 0],
    mode="lines",
    line=dict(color="gray", width=3, dash="dot"),
    showlegend=False,
    name="y-axis trace"
)

# Checkbox to show/hide the surface
show_surface = st.checkbox("Show surface", value=True)

# Build the data list based on the checkbox
data3d = (
    [surface] if show_surface else []
) + [trace_x, trace_y, tan_x, tan_y, pt, pt_xy, dotted_line, xvec, yvec, gradvec, dotted_x, dotted_y] + axes

layout3d = go.Layout(
    scene=dict(
        xaxis=dict(range=[-10, 10], showgrid=False, showbackground=False, zeroline=False, showline=False, showticklabels=False),
        yaxis=dict(range=[-10, 10], showgrid=False, showbackground=False, zeroline=False, showline=False, showticklabels=False),
        zaxis=dict(range=[0, 200], showgrid=False, showbackground=False, zeroline=False, showline=False, showticklabels=False),
    ),
    margin=dict(l=0, r=0, b=0, t=30),
    showlegend=False
)
fig3d = go.Figure(data=data3d, layout=layout3d)

# --- 2D XY-plane plot with gradient vectors ---

import plotly.graph_objs as go2d

# Show 3D plot first
st.plotly_chart(fig3d, use_container_width=True)

# Then show 2D plot below
# Set up the 2D plot
fig2d = go2d.Figure()

# Plot the selected point
fig2d.add_trace(go2d.Scatter(
    x=[x0], y=[y0],
    mode="markers",
    marker=dict(size=12, color="black"),
    name="Selected point"
))

# Plot the x-component vector (red)
fig2d.add_trace(go2d.Scatter(
    x=[x0, x0 + arrow_scale * grad_x],
    y=[y0, y0],
    mode="lines+markers",
    line=dict(color="red", width=5),
    marker=dict(size=[6, 10], color="red"),
    name="Gradient x-component"
))

# Plot the y-component vector (blue)
fig2d.add_trace(go2d.Scatter(
    x=[x0, x0],
    y=[y0, y0 + arrow_scale * grad_y],
    mode="lines+markers",
    line=dict(color="blue", width=5),
    marker=dict(size=[6, 10], color="blue"),
    name="Gradient y-component"
))

# Plot the full gradient vector (magenta)
fig2d.add_trace(go2d.Scatter(
    x=[x0, x0 + arrow_scale * grad_x],
    y=[y0, y0 + arrow_scale * grad_y],
    mode="lines+markers",
    line=dict(color="magenta", width=5),
    marker=dict(size=[6, 10], color="magenta"),
    name="Gradient vector"
))

# Solid x-axis (horizontal at y=0)
fig2d.add_trace(go2d.Scatter(
    x=[x[0], x[-1]],
    y=[0, 0],
    mode="lines",
    line=dict(color="black", width=2),
    showlegend=False
))
# Solid y-axis (vertical at x=0)
fig2d.add_trace(go2d.Scatter(
    x=[0, 0],
    y=[y[0], y[-1]],
    mode="lines",
    line=dict(color="black", width=2),
    showlegend=False
))

fig2d.update_layout(
    xaxis=dict(range=[-10, 10], zeroline=False, showgrid=False, showticklabels=True, title="x"),
    yaxis=dict(range=[-10, 10], zeroline=False, showgrid=False, showticklabels=True, title="y"),
    width=350, height=350,
    margin=dict(l=10, r=10, b=10, t=30),
    showlegend=False,
)

fig2d.update_yaxes(scaleanchor="x", scaleratio=1)

st.plotly_chart(fig2d, use_container_width=True)

st.latex(r"z = x^2 + y^2")
st.markdown(
    r"""
    - The **red curve** is the trace for $y = y_0$ (varying $x$).
    - The **blue curve** is the trace for $x = x_0$ (varying $y$).
    - The **orange line** is the tangent to the $x$-trace at $(x_0, y_0)$.
    - The **green line** is the tangent to the $y$-trace at $(x_0, y_0)$.
    - The **purple point** is the projection of $(x_0, y_0, z_0)$ onto the $xy$-plane.
    - The **magenta arrow** is the gradient vector $\nabla z = (2x_0, 2y_0)$ based at $(x_0, y_0, 0)$.
    """
)
st.latex(r"\frac{\partial z}{\partial x} = 2x \qquad \frac{\partial z}{\partial y} = 2y")
st.latex(rf"\left.\frac{{\partial z}}{{\partial x}}\right|_{{(x_0, y_0)}} = 2 \times {x0:.2f} = {dz_dx:.2f}")
st.latex(rf"\left.\frac{{\partial z}}{{\partial y}}\right|_{{(x_0, y_0)}} = 2 \times {y0:.2f} = {dz_dy:.2f}")

st.subheader("Example 2")

st.markdown(
    r"""
    Now let's explore the **gradient** of a flat Gaussian surface:
    $$
    z = \exp\left(-\frac{(x^2 + y^2)}{2\sigma^2}\right)
    $$
    with $\sigma = 5$.
    """
)

# Parameters for the Gaussian
sigma = 5.0
A = 1.0
xg = np.linspace(-10, 10, 60)
yg = np.linspace(-10, 10, 60)
Xg, Yg = np.meshgrid(xg, yg)
Zg = A * np.exp(-(Xg**2 + Yg**2) / (2 * sigma**2))

# Student selects a point (x0g, y0g)
colxg, colyg = st.columns(2)
with colxg:
    x0g = st.slider("Select $x_0$ (Gaussian)", float(xg[0]), float(xg[-1]), 2.0, 0.1, key="gauss_x0")
with colyg:
    y0g = st.slider("Select $y_0$ (Gaussian)", float(yg[0]), float(yg[-1]), 3.0, 0.1, key="gauss_y0")
z0g = A * np.exp(-(x0g**2 + y0g**2) / (2 * sigma**2))

# x-trace: y = y0g, vary x
xg_trace = np.linspace(-10, 10, 200)
yg_trace_x = np.full_like(xg_trace, y0g)
zg_trace_x = A * np.exp(-(xg_trace**2 + y0g**2) / (2 * sigma**2))

# y-trace: x = x0g, vary y
yg_trace = np.linspace(-10, 10, 200)
xg_trace_y = np.full_like(yg_trace, x0g)
zg_trace_y = A * np.exp(-(x0g**2 + yg_trace**2) / (2 * sigma**2))

# Tangents at (x0g, y0g)
dz_dx_g = -x0g / (sigma**2) * z0g
dz_dy_g = -y0g / (sigma**2) * z0g
xg_tan = np.linspace(x0g - 5, x0g + 5, 100)
zg_tan_x = z0g + dz_dx_g * (xg_tan - x0g)
yg_tan_x = np.full_like(xg_tan, y0g)
yg_tan = np.linspace(y0g - 5, y0g + 5, 100)
zg_tan_y = z0g + dz_dy_g * (yg_tan - y0g)
xg_tan_y = np.full_like(yg_tan, x0g)

# 3D plot for Gaussian
surface_g = go.Surface(x=xg, y=yg, z=Zg, colorscale="Viridis", opacity=0.7, showscale=False)
trace_xg = go.Scatter3d(
    x=xg_trace, y=yg_trace_x, z=zg_trace_x,
    mode="lines", line=dict(color="red", width=5), name="x-trace"
)
trace_yg = go.Scatter3d(
    x=xg_trace_y, y=yg_trace, z=zg_trace_y,
    mode="lines", line=dict(color="blue", width=5), name="y-trace"
)
tan_xg = go.Scatter3d(
    x=xg_tan, y=yg_tan_x, z=zg_tan_x,
    mode="lines", line=dict(color="orange", width=6), name="Tangent to x-trace"
)
tan_yg = go.Scatter3d(
    x=xg_tan_y, y=yg_tan, z=zg_tan_y,
    mode="lines", line=dict(color="green", width=6), name="Tangent to y-trace"
)
ptg = go.Scatter3d(
    x=[x0g], y=[y0g], z=[z0g],
    mode="markers", marker=dict(size=8, color="black"), name="Selected point"
)
axes_g = [
    go.Scatter3d(x=[-10, 10], y=[0, 0], z=[0, 0], mode='lines', line=dict(color='black', width=4)),
    go.Scatter3d(x=[0, 0], y=[-10, 10], z=[0, 0], mode='lines', line=dict(color='black', width=4)),
    go.Scatter3d(x=[0, 0], y=[0, 0], z=[0, 1], mode='lines', line=dict(color='black', width=4))
]
ptg_xy = go.Scatter3d(
    x=[x0g], y=[y0g], z=[0],
    mode="markers",
    marker=dict(size=8, color="purple", symbol="circle"),
    name="Projection in xy-plane"
)
# Gradient vector at (x0g, y0g, 0)
# --- SCALE UP THE GRADIENT VECTORS FOR VISUALIZATION ---
grad_scale_g = 2.5  # (if you use a cone, not currently used)
arrow_scale_g = 50 # <<--- INCREASED FROM 1.2 TO 8.0 FOR VISIBILITY

grad_xg = dz_dx_g
grad_yg = dz_dy_g

# X component vector: (dz_dx_g, 0)
xvecg_end = (x0g + arrow_scale_g * grad_xg, y0g, 0)
xvecg = go.Scatter3d(
    x=[x0g, xvecg_end[0]],
    y=[y0g, xvecg_end[1]],
    z=[0, 0],
    mode="lines+markers",
    line=dict(color="red", width=7),
    marker=dict(size=[4, 10], color="red"),
    name="Gradient x-component",
    showlegend=False
)
# Y component vector: (0, dz_dy_g)
yvecg_end = (x0g, y0g + arrow_scale_g * grad_yg, 0)
yvecg = go.Scatter3d(
    x=[x0g, yvecg_end[0]],
    y=[y0g, yvecg_end[1]],
    z=[0, 0],
    mode="lines+markers",
    line=dict(color="blue", width=7),
    marker=dict(size=[4, 10], color="blue"),
    name="Gradient y-component",
    showlegend=False
)
# Full gradient vector: (dz_dx_g, dz_dy_g)
gradg_end = (x0g + arrow_scale_g * grad_xg, y0g + arrow_scale_g * grad_yg, 0)
gradvecg = go.Scatter3d(
    x=[x0g, gradg_end[0]],
    y=[y0g, gradg_end[1]],
    z=[0, 0],
    mode="lines+markers",
    line=dict(color="magenta", width=7),
    marker=dict(size=[4, 10], color="magenta"),
    name="Gradient vector",
    showlegend=False
)
dotted_lineg = go.Scatter3d(
    x=[x0g, x0g],
    y=[y0g, y0g],
    z=[z0g, 0],
    mode="lines",
    line=dict(color="purple", width=4, dash="dot"),
    showlegend=False,
    name="Projection line"
)
dotted_xg = go.Scatter3d(
    x=[xg[0], xg[-1]],
    y=[y0g, y0g],
    z=[0, 0],
    mode="lines",
    line=dict(color="gray", width=3, dash="dot"),
    showlegend=False,
    name="x-axis trace"
)
dotted_yg = go.Scatter3d(
    x=[x0g, x0g],
    y=[yg[0], yg[-1]],
    z=[0, 0],
    mode="lines",
    line=dict(color="gray", width=3, dash="dot"),
    showlegend=False,
    name="y-axis trace"
)
show_surface_g = st.checkbox("Show Gaussian surface", value=True)
data3dg = (
    [surface_g] if show_surface_g else []
) + [trace_xg, trace_yg, tan_xg, tan_yg, ptg, ptg_xy, dotted_lineg, xvecg, yvecg, gradvecg, dotted_xg, dotted_yg] + axes_g
layout3dg = go.Layout(
    scene=dict(
        xaxis=dict(range=[-10, 10], showgrid=False, showbackground=False, zeroline=False, showline=False, showticklabels=False),
        yaxis=dict(range=[-10, 10], showgrid=False, showbackground=False, zeroline=False, showline=False, showticklabels=False),
        zaxis=dict(range=[0, 1.1], showgrid=False, showbackground=False, zeroline=False, showline=False, showticklabels=False),
    ),
    margin=dict(l=0, r=0, b=0, t=30),
    showlegend=False
)
fig3dg = go.Figure(data=data3dg, layout=layout3dg)
st.plotly_chart(fig3dg, use_container_width=True)

# 2D plot for Gaussian
fig2dg = go2d.Figure()
fig2dg.add_trace(go2d.Scatter(
    x=[x0g], y=[y0g],
    mode="markers",
    marker=dict(size=12, color="black"),
    name="Selected point"
))
fig2dg.add_trace(go2d.Scatter(
    x=[x0g, x0g + arrow_scale_g * grad_xg],
    y=[y0g, y0g],
    mode="lines+markers",
    line=dict(color="red", width=5),
    marker=dict(size=[6, 10], color="red"),
    name="Gradient x-component"
))
fig2dg.add_trace(go2d.Scatter(
    x=[x0g, x0g],
    y=[y0g, y0g + arrow_scale_g * grad_yg],
    mode="lines+markers",
    line=dict(color="blue", width=5),
    marker=dict(size=[6, 10], color="blue"),
    name="Gradient y-component"
))
fig2dg.add_trace(go2d.Scatter(
    x=[x0g, x0g + arrow_scale_g * grad_xg],
    y=[y0g, y0g + arrow_scale_g * grad_yg],
    mode="lines+markers",
    line=dict(color="magenta", width=5),
    marker=dict(size=[6, 10], color="magenta"),
    name="Gradient vector"
))
fig2dg.add_trace(go2d.Scatter(
    x=[xg[0], xg[-1]],
    y=[0, 0],
    mode="lines",
    line=dict(color="black", width=2),
    showlegend=False
))
fig2dg.add_trace(go2d.Scatter(
    x=[0, 0],
    y=[yg[0], yg[-1]],
    mode="lines",
    line=dict(color="black", width=2),
    showlegend=False
))
fig2dg.update_layout(
    xaxis=dict(range=[-10, 10], zeroline=False, showgrid=False, showticklabels=True, title="x"),
    yaxis=dict(range=[-10, 10], zeroline=False, showgrid=False, showticklabels=True, title="y"),
    width=350, height=350,
    margin=dict(l=10, r=10, b=10, t=30),
    showlegend=False,
)
fig2dg.update_yaxes(scaleanchor="x", scaleratio=1)
st.plotly_chart(fig2dg, use_container_width=True)

st.latex(r"z = \exp\left(-\frac{x^2 + y^2}{2\sigma^2}\right),\quad \sigma=5")
st.markdown(
    r"""
    - The **red curve** is the trace for $y = y_0$ (varying $x$).
    - The **blue curve** is the trace for $x = x_0$ (varying $y$).
    - The **orange line** is the tangent to the $x$-trace at $(x_0, y_0)$.
    - The **green line** is the tangent to the $y$-trace at $(x_0, y_0)$.
    - The **purple point** is the projection of $(x_0, y_0, z_0)$ onto the $xy$-plane.
    - The **magenta arrow** is the gradient vector $\nabla z$ based at $(x_0, y_0, 0)$.

    **Note:** For visualization purposes, the gradient vectors are scaled up so they are visible. The actual gradients for the Gaussian surface are much smaller in magnitude.
    """
)
st.latex(r"\frac{\partial z}{\partial x} = -\frac{x}{\sigma^2}z \qquad \frac{\partial z}{\partial y} = -\frac{y}{\sigma^2}z")
st.latex(rf"\left.\frac{{\partial z}}{{\partial x}}\right|_{{(x_0, y_0)}} = -\frac{{{x0g:.2f}}}{{{sigma:.2f}^2}} \times {z0g:.2f} = {dz_dx_g:.3f}")
st.latex(rf"\left.\frac{{\partial z}}{{\partial y}}\right|_{{(x_0, y_0)}} = -\frac{{{y0g:.2f}}}{{{sigma:.2f}^2}} \times {z0g:.2f} = {dz_dy_g:.3f}")

st.subheader("Homework")

st.markdown("**Instructions:** For each surface below:\n"
            "1. Find the gradient vector $\\nabla z = \\left(\\frac{\\partial z}{\\partial x}, \\frac{\\partial z}{\\partial y}\\right)$.\n"
            "2. Evaluate the gradient at the given point.\n\n---")

# Problem 1
st.markdown(r"""
**Problem 1:**  
$$
z = x^2 - 3xy + 2y^2
$$
at $(x, y) = (1, 2)$
""")
with st.expander("Show Solution 1"):
    st.latex(r"""
z = x^2 - 3xy + 2y^2 \\
\frac{\partial z}{\partial x} = 2x - 3y \\
\frac{\partial z}{\partial y} = -3x + 4y \\
\nabla z = (2x - 3y,\, -3x + 4y) \\
\nabla z|_{(1,2)} = (2 \times 1 - 3 \times 2,\, -3 \times 1 + 4 \times 2) = (2 - 6,\, -3 + 8) = (-4,\, 5)
""")

# Problem 2
st.markdown(r"""
**Problem 2:**  
$$
z = \sin(x)\cos(y)
$$
at $(x, y) = (0, \frac{\pi}{2})$
""")
with st.expander("Show Solution 2"):
    st.latex(r"""
z = \sin(x)\cos(y) \\
\frac{\partial z}{\partial x} = \cos(x)\cos(y) \\
\frac{\partial z}{\partial y} = -\sin(x)\sin(y) \\
\nabla z = (\cos(x)\cos(y),\, -\sin(x)\sin(y)) \\
\nabla z|_{(0,\,\frac{\pi}{2})} = (\cos(0)\cos(\frac{\pi}{2}),\, -\sin(0)\sin(\frac{\pi}{2})) = (1 \times 0,\, 0) = (0,\, 0)
""")

# Problem 3
st.markdown(r"""
**Problem 3:**  
$$
z = e^{x^2 + y^2}
$$
at $(x, y) = (0, 0)$
""")
with st.expander("Show Solution 3"):
    st.latex(r"""
z = e^{x^2 + y^2} \\
\frac{\partial z}{\partial x} = 2x\,e^{x^2 + y^2} \\
\frac{\partial z}{\partial y} = 2y\,e^{x^2 + y^2} \\
\nabla z = (2x\,e^{x^2 + y^2},\, 2y\,e^{x^2 + y^2}) \\
\nabla z|_{(0,0)} = (0,\, 0)
""")

# Problem 4
st.markdown(r"""
**Problem 4:**  
$$
z = \ln(x^2 + y^2 + 1)
$$
at $(x, y) = (1, -1)$
""")
with st.expander("Show Solution 4"):
    st.latex(r"""
z = \ln(x^2 + y^2 + 1) \\
\frac{\partial z}{\partial x} = \frac{2x}{x^2 + y^2 + 1} \\
\frac{\partial z}{\partial y} = \frac{2y}{x^2 + y^2 + 1} \\
\nabla z = \left(\frac{2x}{x^2 + y^2 + 1},\, \frac{2y}{x^2 + y^2 + 1}\right) \\
\nabla z|_{(1,-1)} = \left(\frac{2 \times 1}{1^2 + (-1)^2 + 1},\, \frac{2 \times (-1)}{1^2 + (-1)^2 + 1}\right) = \left(\frac{2}{3},\, -\frac{2}{3}\right)
""")

# Problem 5
st.markdown(r"""
**Problem 5:**  
$$
z = x^3y - y^3x
$$
at $(x, y) = (1, 1)$
""")
with st.expander("Show Solution 5"):
    st.latex(r"""
z = x^3y - y^3x \\
\frac{\partial z}{\partial x} = 3x^2y - y^3 \\
\frac{\partial z}{\partial y} = x^3 - 3y^2x \\
\nabla z = (3x^2y - y^3,\, x^3 - 3y^2x) \\
\nabla z|_{(1,1)} = (3 \times 1^2 \times 1 - 1^3,\, 1^3 - 3 \times 1^2 \times 1) = (3 - 1,\, 1 - 3) = (2,\, -2)
""")

# Problem 6
st.markdown(r"""
**Problem 6:**  
$$
z = x^2y^2
$$
at $(x, y) = (2, -1)$
""")
with st.expander("Show Solution 6"):
    st.latex(r"""
z = x^2y^2 \\
\frac{\partial z}{\partial x} = 2x y^2 \\
\frac{\partial z}{\partial y} = 2y x^2 \\
\nabla z = (2x y^2,\, 2y x^2) \\
\nabla z|_{(2,-1)} = (2 \times 2 \times (-1)^2,\, 2 \times (-1) \times 2^2) = (4,\, -8)
""")

# Problem 7
st.markdown(r"""
**Problem 7:**  
$$
z = \arctan\left(\frac{y}{x}\right)
$$
at $(x, y) = (1, 1)$
""")
with st.expander("Show Solution 7"):
    st.latex(r"""
z = \arctan\left(\frac{y}{x}\right) \\
\frac{\partial z}{\partial x} = \frac{-y}{x^2 + y^2} \\
\frac{\partial z}{\partial y} = \frac{x}{x^2 + y^2} \\
\nabla z = \left(\frac{-y}{x^2 + y^2},\, \frac{x}{x^2 + y^2}\right) \\
\nabla z|_{(1,1)} = \left(\frac{-1}{2},\, \frac{1}{2}\right)
""")

# Problem 8
st.markdown(r"""
**Problem 8:**  
$$
z = \sqrt{x^2 + 4y^2}
$$
at $(x, y) = (3, 1)$
""")
with st.expander("Show Solution 8"):
    st.latex(r"""
z = \sqrt{x^2 + 4y^2} \\
\frac{\partial z}{\partial x} = \frac{x}{\sqrt{x^2 + 4y^2}} \\
\frac{\partial z}{\partial y} = \frac{4y}{\sqrt{x^2 + 4y^2}} \\
\nabla z = \left(\frac{x}{\sqrt{x^2 + 4y^2}},\, \frac{4y}{\sqrt{x^2 + 4y^2}}\right) \\
\nabla z|_{(3,1)} = \left(\frac{3}{\sqrt{3^2 + 4 \times 1^2}},\, \frac{4 \times 1}{\sqrt{3^2 + 4 \times 1^2}}\right) = \left(\frac{3}{\sqrt{13}},\, \frac{4}{\sqrt{13}}\right)
""")

# Problem 9
st.markdown(r"""
**Problem 9:**  
$$
z = e^{xy}
$$
at $(x, y) = (0, 2)$
""")
with st.expander("Show Solution 9"):
    st.latex(r"""
z = e^{xy} \\
\frac{\partial z}{\partial x} = y e^{xy} \\
\frac{\partial z}{\partial y} = x e^{xy} \\
\nabla z = (y e^{xy},\, x e^{xy}) \\
\nabla z|_{(0,2)} = (2 e^{0},\, 0 \times e^{0}) = (2,\, 0)
""")

# Problem 10
st.markdown(r"""
**Problem 10:**  
$$
z = \frac{x}{y}
$$
at $(x, y) = (2, 1)$
""")
with st.expander("Show Solution 10"):
    st.latex(r"""
z = \frac{x}{y} \\
\frac{\partial z}{\partial x} = \frac{1}{y} \\
\frac{\partial z}{\partial y} = -\frac{x}{y^2} \\
\nabla z = \left(\frac{1}{y},\, -\frac{x}{y^2}\right) \\
\nabla z|_{(2,1)} = (1,\, -2)
""")