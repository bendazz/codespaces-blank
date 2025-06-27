import streamlit as st
import numpy as np
import plotly.graph_objs as go

st.title("Partial Derivatives and Tangent Lines to Traces")

st.subheader("Example 1")

st.markdown(
    r"""
    This interactive plot lets you explore **traces** and **tangent lines** of the surface $z = x^2 + y^2$.
    - Choose to fix either $x$ or $y$.
    - Select the value at which to fix the variable (this determines the trace).
    - Select a value for the other variable (this determines the point of tangency).
    - The tangent line to the trace at that point will be drawn.
    """
)

# Surface grid
x = np.linspace(-10, 10, 60)
y = np.linspace(-10, 10, 60)
X, Y = np.meshgrid(x, y)
Z = X**2 + Y**2

# Controls
fix_var = st.radio("Which variable do you want to fix?", ("x", "y"), horizontal=True)
if fix_var == "x":
    fixed_val = st.slider("Select the value of x to fix", float(x[0]), float(x[-1]), 0.0, 0.1)
    var_val = st.slider("Select the value of y for the tangent point", float(y[0]), float(y[-1]), 0.0, 0.1)
else:
    fixed_val = st.slider("Select the value of y to fix", float(y[0]), float(y[-1]), 0.0, 0.1)
    var_val = st.slider("Select the value of x for the tangent point", float(x[0]), float(x[-1]), 0.0, 0.1)

# Prepare trace and tangent line
if fix_var == "x":
    y_trace = np.linspace(-10, 10, 200)
    x_trace = np.full_like(y_trace, fixed_val)
    z_trace = fixed_val**2 + y_trace**2

    # Point of tangency
    x0, y0 = fixed_val, var_val
    z0 = x0**2 + y0**2
    dz_dy = 2 * y0  # derivative of z wrt y, with x fixed

    # Tangent line in y (make it longer: ±5 instead of ±2)
    y_tan = np.linspace(y0 - 5, y0 + 5, 100)
    z_tan = z0 + dz_dy * (y_tan - y0)
    x_tan = np.full_like(y_tan, x0)
else:
    x_trace = np.linspace(-10, 10, 200)
    y_trace = np.full_like(x_trace, fixed_val)
    z_trace = x_trace**2 + fixed_val**2

    # Point of tangency
    x0, y0 = var_val, fixed_val
    z0 = x0**2 + y0**2
    dz_dx = 2 * x0  # derivative of z wrt x, with y fixed

    # Tangent line in x (make it longer: ±5 instead of ±2)
    x_tan = np.linspace(x0 - 5, x0 + 5, 100)
    z_tan = z0 + dz_dx * (x_tan - x0)
    y_tan = np.full_like(x_tan, y0)

# Add checkboxes to show/hide the surface and the vertical plane
show_surface = st.checkbox("Show surface", value=True)
show_plane = st.checkbox("Show vertical plane", value=True)

# 3D plot components
surface = go.Surface(x=x, y=y, z=Z, colorscale="Viridis", opacity=0.7, showscale=False)
trace3d = go.Scatter3d(
    x=x_trace, y=y_trace, z=z_trace,
    mode="lines", line=dict(color="red", width=6), name="Trace"
)
tangent3d = go.Scatter3d(
    x=x_tan, y=y_tan, z=z_tan,
    mode="lines", line=dict(color="orange", width=6), name="Tangent line"
)
point3d = go.Scatter3d(
    x=[x0], y=[y0], z=[z0],
    mode="markers", marker=dict(size=8, color="black"), name="Point of tangency"
)
axes = [
    go.Scatter3d(x=[-10, 10], y=[0, 0], z=[0, 0], mode='lines', line=dict(color='black', width=5)),
    go.Scatter3d(x=[0, 0], y=[-10, 10], z=[0, 0], mode='lines', line=dict(color='black', width=5)),
    go.Scatter3d(x=[0, 0], y=[0, 0], z=[0, 200], mode='lines', line=dict(color='black', width=5))
]
plane_x = [-10, 10, 10, -10]
plane_y = [-10, -10, 10, 10]
plane_z = [0, 0, 0, 0]
xy_plane = go.Mesh3d(
    x=plane_x, y=plane_y, z=plane_z,
    color='lightblue', opacity=0.3, showscale=False
)

# Add a lightly colored vertical plane at the fixed value (as a Surface)
if show_plane:
    if fix_var == "x":
        y_plane = np.linspace(-10, 10, 2)
        z_plane = np.linspace(0, 200, 2)
        Y_plane, Z_plane = np.meshgrid(y_plane, z_plane)
        X_plane = np.full_like(Y_plane, fixed_val)
        vertical_plane = go.Surface(
            x=X_plane,
            y=Y_plane,
            z=Z_plane,
            showscale=False,
            opacity=0.25,
            colorscale=[[0, 'orange'], [1, 'orange']],
            name='x=const plane',
            hoverinfo='skip'
        )
    else:
        x_plane = np.linspace(-10, 10, 2)
        z_plane = np.linspace(0, 200, 2)
        X_plane, Z_plane = np.meshgrid(x_plane, z_plane)
        Y_plane = np.full_like(X_plane, fixed_val)
        vertical_plane = go.Surface(
            x=X_plane,
            y=Y_plane,
            z=Z_plane,
            showscale=False,
            opacity=0.25,
            colorscale=[[0, 'orange'], [1, 'orange']],
            name='y=const plane',
            hoverinfo='skip'
        )
else:
    vertical_plane = None

# Prepare data list based on checkboxes
data3d = axes + [trace3d, tangent3d, point3d, xy_plane]
if show_surface:
    data3d.insert(3, surface)  # Insert surface before xy_plane for better layering
if show_plane and vertical_plane is not None:
    data3d.append(vertical_plane)

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

# 2D plot of trace and tangent
if fix_var == "x":
    ind_var = y_trace
    ind_label = "y"
    title_2d = f"Trace at x = {fixed_val:.2f}: y vs z"
    plane_color = "rgba(255,165,0,0.10)"
    trace_eq = rf"z = {fixed_val:.2f}^2 + y^2"
    # Tangent line in 2D
    tan2d = go.Scatter(
        x=y_tan, y=z_tan, mode="lines",
        line=dict(color="orange", width=4), name="Tangent line"  # removed dash="dash"
    )
    pt2d = go.Scatter(
        x=[y0], y=[z0], mode="markers",
        marker=dict(size=8, color="black"), name="Point of tangency"
    )
    trace2d = go.Scatter(
        x=ind_var, y=z_trace, mode="lines",
        line=dict(color="red", width=4), name="Trace"
    )
else:
    ind_var = x_trace
    ind_label = "x"
    title_2d = f"Trace at y = {fixed_val:.2f}: x vs z"
    plane_color = "rgba(255,165,0,0.10)"
    trace_eq = rf"z = x^2 + {fixed_val:.2f}^2"
    tan2d = go.Scatter(
        x=x_tan, y=z_tan, mode="lines",
        line=dict(color="orange", width=4), name="Tangent line"  # removed dash="dash"
    )
    pt2d = go.Scatter(
        x=[x0], y=[z0], mode="markers",
        marker=dict(size=8, color="black"), name="Point of tangency"
    )
    trace2d = go.Scatter(
        x=ind_var, y=z_trace, mode="lines",
        line=dict(color="red", width=4), name="Trace"
    )

axis_x = go.Scatter(
    x=[-10, 10], y=[0, 0], mode="lines",
    line=dict(color="black", width=3), showlegend=False, hoverinfo="skip"
)
axis_z = go.Scatter(
    x=[0, 0], y=[0, 200], mode="lines",
    line=dict(color="black", width=3), showlegend=False, hoverinfo="skip"
)
layout2d = go.Layout(
    title=title_2d,
    xaxis=dict(title=ind_label, range=[-10, 10], showgrid=False, showticklabels=False, showline=False, zeroline=False),
    yaxis=dict(title="z", range=[0, 200], showgrid=False, showticklabels=False, showline=False, zeroline=False),
    plot_bgcolor=plane_color,
    margin=dict(l=40, r=10, b=40, t=40),
    showlegend=False
)
fig2d = go.Figure(data=[axis_x, axis_z, trace2d, tan2d, pt2d], layout=layout2d)

col3d, col2d = st.columns(2)
with col3d:
    st.plotly_chart(fig3d, use_container_width=True)
    st.latex(r"z = x^2 + y^2")
    if fix_var == "x":
        st.latex(r"\frac{\partial z}{\partial y} = 2y")
        st.latex(rf"\left.\frac{{\partial z}}{{\partial y}}\right|_{{y={y0:.2f}}} = 2 \times {y0:.2f} = {dz_dy:.2f}")
    else:
        st.latex(r"\frac{\partial z}{\partial x} = 2x")
        st.latex(rf"\left.\frac{{\partial z}}{{\partial x}}\right|_{{x={x0:.2f}}} = 2 \times {x0:.2f} = {dz_dx:.2f}")
with col2d:
    st.plotly_chart(fig2d, use_container_width=True)
    st.latex(trace_eq)
    if fix_var == "x":
        st.latex(r"\frac{dz}{dy} = 2y")
        st.latex(rf"\left.\frac{{dz}}{{dy}}\right|_{{y={y0:.2f}}} = 2 \times {y0:.2f} = {dz_dy:.2f}")
    else:
        st.latex(r"\frac{dz}{dx} = 2x")
        st.latex(rf"\left.\frac{{dz}}{{dx}}\right|_{{x={x0:.2f}}} = 2 \times {x0:.2f} = {dz_dx:.2f}")

st.markdown("---")
st.subheader("Example 2: Gaussian Bump")

st.markdown(
    r"""
    Now explore traces and tangent lines for the surface
    $$
    z = e^{-\frac{x^2 + y^2}{2\sigma^2}}
    $$
    with a large $\sigma$ (wide bump).
    """
)

# Parameters for the Gaussian
sigma = 5.0  # Large standard deviation for a wide bump

# Surface grid for Gaussian
xg = np.linspace(-15, 15, 60)
yg = np.linspace(-15, 15, 60)
Xg, Yg = np.meshgrid(xg, yg)
Zg = np.exp(-(Xg**2 + Yg**2) / (2 * sigma**2))

# Controls for Gaussian
fix_var_g = st.radio("Which variable do you want to fix? (Gaussian)", ("x", "y"), horizontal=True, key="gauss_fix_var")
if fix_var_g == "x":
    fixed_val_g = st.slider("Select the value of x to fix (Gaussian)", float(xg[0]), float(xg[-1]), 0.0, 0.1, key="gauss_fixed_x")
    var_val_g = st.slider("Select the value of y for the tangent point (Gaussian)", float(yg[0]), float(yg[-1]), 0.0, 0.1, key="gauss_var_y")
else:
    fixed_val_g = st.slider("Select the value of y to fix (Gaussian)", float(yg[0]), float(yg[-1]), 0.0, 0.1, key="gauss_fixed_y")
    var_val_g = st.slider("Select the value of x for the tangent point (Gaussian)", float(xg[0]), float(xg[-1]), 0.0, 0.1, key="gauss_var_x")

# Prepare trace and tangent line for Gaussian
if fix_var_g == "x":
    y_trace_g = np.linspace(-15, 15, 200)
    x_trace_g = np.full_like(y_trace_g, fixed_val_g)
    z_trace_g = np.exp(-(fixed_val_g**2 + y_trace_g**2) / (2 * sigma**2))

    # Point of tangency
    x0g, y0g = fixed_val_g, var_val_g
    z0g = np.exp(-(x0g**2 + y0g**2) / (2 * sigma**2))
    dz_dy_g = -y0g / (sigma**2) * z0g  # derivative wrt y, x fixed

    # Tangent line in y
    y_tan_g = np.linspace(y0g - 5, y0g + 5, 100)
    z_tan_g = z0g + dz_dy_g * (y_tan_g - y0g)
    x_tan_g = np.full_like(y_tan_g, x0g)
else:
    x_trace_g = np.linspace(-15, 15, 200)
    y_trace_g = np.full_like(x_trace_g, fixed_val_g)
    z_trace_g = np.exp(-(x_trace_g**2 + fixed_val_g**2) / (2 * sigma**2))

    # Point of tangency
    x0g, y0g = var_val_g, fixed_val_g
    z0g = np.exp(-(x0g**2 + y0g**2) / (2 * sigma**2))
    dz_dx_g = -x0g / (sigma**2) * z0g  # derivative wrt x, y fixed

    # Tangent line in x
    x_tan_g = np.linspace(x0g - 5, x0g + 5, 100)
    z_tan_g = z0g + dz_dx_g * (x_tan_g - x0g)
    y_tan_g = np.full_like(x_tan_g, y0g)

# Add checkboxes to show/hide the surface and the vertical plane for Gaussian
show_surface_g = st.checkbox("Show surface (Gaussian)", value=True, key="gauss_show_surface")
show_plane_g = st.checkbox("Show vertical plane (Gaussian)", value=True, key="gauss_show_plane")

# 3D plot components for Gaussian
surface_g = go.Surface(x=xg, y=yg, z=Zg, colorscale="Viridis", opacity=0.7, showscale=False)
trace3d_g = go.Scatter3d(
    x=x_trace_g, y=y_trace_g, z=z_trace_g,
    mode="lines", line=dict(color="red", width=6), name="Trace"
)
tangent3d_g = go.Scatter3d(
    x=x_tan_g, y=y_tan_g, z=z_tan_g,
    mode="lines", line=dict(color="orange", width=6), name="Tangent line"
)
point3d_g = go.Scatter3d(
    x=[x0g], y=[y0g], z=[z0g],
    mode="markers", marker=dict(size=8, color="black"), name="Point of tangency"
)
axes_g = [
    go.Scatter3d(x=[-15, 15], y=[0, 0], z=[0, 0], mode='lines', line=dict(color='black', width=5)),
    go.Scatter3d(x=[0, 0], y=[-15, 15], z=[0, 0], mode='lines', line=dict(color='black', width=5)),
    go.Scatter3d(x=[0, 0], y=[0, 0], z=[0, 1.1], mode='lines', line=dict(color='black', width=5))
]
plane_xg = [-15, 15, 15, -15]
plane_yg = [-15, -15, 15, 15]
plane_zg = [0, 0, 0, 0]
xy_plane_g = go.Mesh3d(
    x=plane_xg, y=plane_yg, z=plane_zg,
    color='lightblue', opacity=0.3, showscale=False
)

# Add a lightly colored vertical plane at the fixed value (as a Surface) for Gaussian
if show_plane_g:
    if fix_var_g == "x":
        y_plane_g = np.linspace(-15, 15, 2)
        z_plane_g = np.linspace(0, 1.1, 2)
        Y_plane_g, Z_plane_g = np.meshgrid(y_plane_g, z_plane_g)
        X_plane_g = np.full_like(Y_plane_g, fixed_val_g)
        vertical_plane_g = go.Surface(
            x=X_plane_g,
            y=Y_plane_g,
            z=Z_plane_g,
            showscale=False,
            opacity=0.25,
            colorscale=[[0, 'orange'], [1, 'orange']],
            name='x=const plane',
            hoverinfo='skip'
        )
    else:
        x_plane_g = np.linspace(-15, 15, 2)
        z_plane_g = np.linspace(0, 1.1, 2)
        X_plane_g, Z_plane_g = np.meshgrid(x_plane_g, z_plane_g)
        Y_plane_g = np.full_like(X_plane_g, fixed_val_g)
        vertical_plane_g = go.Surface(
            x=X_plane_g,
            y=Y_plane_g,
            z=Z_plane_g,
            showscale=False,
            opacity=0.25,
            colorscale=[[0, 'orange'], [1, 'orange']],
            name='y=const plane',
            hoverinfo='skip'
        )
else:
    vertical_plane_g = None

# Prepare data list based on checkboxes for Gaussian
data3d_g = axes_g + [trace3d_g, tangent3d_g, point3d_g, xy_plane_g]
if show_surface_g:
    data3d_g.insert(3, surface_g)
if show_plane_g and vertical_plane_g is not None:
    data3d_g.append(vertical_plane_g)

layout3d_g = go.Layout(
    scene=dict(
        xaxis=dict(range=[-15, 15], showgrid=False, showbackground=False, zeroline=False, showline=False, showticklabels=False),
        yaxis=dict(range=[-15, 15], showgrid=False, showbackground=False, zeroline=False, showline=False, showticklabels=False),
        zaxis=dict(range=[0, 1.1], showgrid=False, showbackground=False, zeroline=False, showline=False, showticklabels=False),
    ),
    margin=dict(l=0, r=0, b=0, t=30),
    showlegend=False
)
fig3d_g = go.Figure(data=data3d_g, layout=layout3d_g)

# 2D plot of trace and tangent for Gaussian
if fix_var_g == "x":
    ind_var_g = y_trace_g
    ind_label_g = "y"
    title_2d_g = f"Trace at x = {fixed_val_g:.2f}: y vs z"
    plane_color_g = "rgba(255,165,0,0.10)"
    trace_eq_g = rf"z = \exp\left(-\frac{{{fixed_val_g:.2f}^2 + y^2}}{{2 \times {sigma:.2f}^2}}\right)"
    # Tangent line in 2D
    tan2d_g = go.Scatter(
        x=y_tan_g, y=z_tan_g, mode="lines",
        line=dict(color="orange", width=4), name="Tangent line"
    )
    pt2d_g = go.Scatter(
        x=[y0g], y=[z0g], mode="markers",
        marker=dict(size=8, color="black"), name="Point of tangency"
    )
    trace2d_g = go.Scatter(
        x=ind_var_g, y=z_trace_g, mode="lines",
        line=dict(color="red", width=4), name="Trace"
    )
else:
    ind_var_g = x_trace_g
    ind_label_g = "x"
    title_2d_g = f"Trace at y = {fixed_val_g:.2f}: x vs z"
    plane_color_g = "rgba(255,165,0,0.10)"
    trace_eq_g = rf"z = \exp\left(-\frac{{x^2 + {fixed_val_g:.2f}^2}}{{2 \times {sigma:.2f}^2}}\right)"
    tan2d_g = go.Scatter(
        x=x_tan_g, y=z_tan_g, mode="lines",
        line=dict(color="orange", width=4), name="Tangent line"  # removed dash="dash"
    )
    pt2d_g = go.Scatter(
        x=[x0g], y=[z0g], mode="markers",
        marker=dict(size=8, color="black"), name="Point of tangency"
    )
    trace2d_g = go.Scatter(
        x=ind_var_g, y=z_trace_g, mode="lines",
        line=dict(color="red", width=4), name="Trace"
    )

axis_x_g = go.Scatter(
    x=[-15, 15], y=[0, 0], mode="lines",
    line=dict(color="black", width=3), showlegend=False, hoverinfo="skip"
)
axis_z_g = go.Scatter(
    x=[0, 0], y=[0, 1.1], mode="lines",
    line=dict(color="black", width=3), showlegend=False, hoverinfo="skip"
)
layout2d_g = go.Layout(
    title=title_2d_g,
    xaxis=dict(title=ind_label_g, range=[-15, 15], showgrid=False, showticklabels=False, showline=False, zeroline=False),
    yaxis=dict(title="z", range=[0, 1.1], showgrid=False, showticklabels=False, showline=False, zeroline=False),
    plot_bgcolor=plane_color_g,
    margin=dict(l=40, r=10, b=40, t=40),
    showlegend=False
)
fig2d_g = go.Figure(data=[axis_x_g, axis_z_g, trace2d_g, tan2d_g, pt2d_g], layout=layout2d_g)

col3d_g, col2d_g = st.columns(2)
with col3d_g:
    st.plotly_chart(fig3d_g, use_container_width=True)
    st.latex(rf"z = \exp\left(-\frac{{x^2 + y^2}}{{2 \times 5^2}}\right)")
    if fix_var_g == "x":
        st.latex(rf"\frac{{\partial z}}{{\partial y}} = -\frac{{y}}{{{sigma:.2f}^2}} z")
        st.latex(rf"\left.\frac{{\partial z}}{{\partial y}}\right|_{{y={y0g:.2f}}} = -\frac{{{y0g:.2f}}}{{{sigma:.2f}^2}} \times {z0g:.3f} = {dz_dy_g:.3f}")
    else:
        st.latex(rf"\frac{{\partial z}}{{\partial x}} = -\frac{{x}}{{{sigma:.2f}^2}} z")
        st.latex(rf"\left.\frac{{\partial z}}{{\partial x}}\right|_{{x={x0g:.2f}}} = -\frac{{{x0g:.2f}}}{{{sigma:.2f}^2}} \times {z0g:.3f} = {dz_dx_g:.3f}")
with col2d_g:
    st.plotly_chart(fig2d_g, use_container_width=True)
    st.latex(trace_eq_g)
    if fix_var_g == "x":
        st.latex(rf"\frac{{dz}}{{dy}} = -\frac{{y}}{{{sigma:.2f}^2}} z")
        st.latex(rf"\left.\frac{{dz}}{{dy}}\right|_{{y={y0g:.2f}}} = -\frac{{{y0g:.2f}}}{{{sigma:.2f}^2}} \times {z0g:.3f} = {dz_dy_g:.3f}")
    else:
        st.latex(rf"\frac{{dz}}{{dx}} = -\frac{{x}}{{{sigma:.2f}^2}} z")
        st.latex(rf"\left.\frac{{dz}}{{dx}}\right|_{{x={x0g:.2f}}} = -\frac{{{x0g:.2f}}}{{{sigma:.2f}^2}} \times {z0g:.3f} = {dz_dx_g:.3f}")


st.subheader("Using SymPy to Compute Partial Derivatives")

st.markdown(
    r"""
You can use the `sympy` library to compute partial derivatives symbolically and evaluate them at specific points.
Let's see how this works for both the paraboloid and the Gaussian bump.
"""
)

import sympy as sp

# Paraboloid example
st.markdown("**Example 1: Paraboloid $z = x^2 + y^2$**")

st.code(
    '''import sympy as sp

x, y = sp.symbols('x y')
z = x**2 + y**2

dz_dx = sp.diff(z, x)
dz_dy = sp.diff(z, y)

# Evaluate at (x, y) = (2, 3)
dz_dx_val = dz_dx.subs({x: 2, y: 3})
dz_dy_val = dz_dy.subs({x: 2, y: 3})
''', language="python"
)

x_sym, y_sym = sp.symbols('x y')
z_parab = x_sym**2 + y_sym**2
dz_dx_parab = sp.diff(z_parab, x_sym)
dz_dy_parab = sp.diff(z_parab, y_sym)
x_val, y_val = 2, 3
dz_dx_val = dz_dx_parab.subs({x_sym: x_val, y_sym: y_val})
dz_dy_val = dz_dy_parab.subs({x_sym: x_val, y_sym: y_val})

st.latex(r"z = x^2 + y^2")
st.latex(r"\frac{\partial z}{\partial x} = " + sp.latex(dz_dx_parab))
st.latex(r"\frac{\partial z}{\partial y} = " + sp.latex(dz_dy_parab))
st.markdown(f"At $x={x_val}$, $y={y_val}$:")
st.latex(rf"\left.\frac{{\partial z}}{{\partial x}}\right|_{{(x, y) = ({x_val}, {y_val})}} = {dz_dx_val}")
st.latex(rf"\left.\frac{{\partial z}}{{\partial y}}\right|_{{(x, y) = ({x_val}, {y_val})}} = {dz_dy_val}")

st.markdown("---")
# Gaussian bump example
st.markdown("**Example 2: Gaussian Bump**")

st.latex(r"z = \exp\left(-\frac{x^2 + y^2}{2 \times 5^2}\right)")

st.code(
    '''import sympy as sp

x, y = sp.symbols('x y')
sigma = 5
z = sp.exp(-(x**2 + y**2) / (2 * sigma**2))

dz_dx = sp.diff(z, x)
dz_dy = sp.diff(z, y)

# Evaluate at (x, y) = (1, -2)
dz_dx_val = dz_dx.subs({x: 1, y: -2})
dz_dy_val = dz_dy.subs({x: 1, y: -2})
''', language="python"
)

sigma_val = 5
xg_val, yg_val = 1, -2
x_sym, y_sym = sp.symbols('x y')
z_gauss = sp.exp(-(x_sym**2 + y_sym**2) / (2 * sigma_val**2))
dz_dx_gauss = sp.diff(z_gauss, x_sym)
dz_dy_gauss = sp.diff(z_gauss, y_sym)
dz_dxg_val = dz_dx_gauss.subs({x_sym: xg_val, y_sym: yg_val}).evalf()
dz_dyg_val = dz_dy_gauss.subs({x_sym: xg_val, y_sym: yg_val}).evalf()

st.latex(r"z = \exp\left(-\frac{x^2 + y^2}{2 \times 5^2}\right)")
st.latex(r"\frac{\partial z}{\partial x} = " + sp.latex(dz_dx_gauss))
st.latex(r"\frac{\partial z}{\partial y} = " + sp.latex(dz_dy_gauss))
st.markdown(f"At $x={xg_val}$, $y={yg_val}$, $\\sigma=5$:")
st.latex(rf"\left.\frac{{\partial z}}{{\partial x}}\right|_{{(x, y) = ({xg_val}, {yg_val})}} = {dz_dxg_val:.4f}")
st.latex(rf"\left.\frac{{\partial z}}{{\partial y}}\right|_{{(x, y) = ({xg_val}, {yg_val})}} = {dz_dyg_val:.4f}")



st.markdown("---")
st.header("Homework")

st.markdown(
    r"""
For each problem below, find the indicated partial derivative and then evaluate it at the given point.  
Click "Solution" to check your work.
"""
)

# --- Problem 1 ---
st.markdown(r"""
**Problem 1**  
Surface: $z = 3x^2y + 2y^3$  
Find $\frac{\partial z}{\partial x}$ and evaluate at $(x, y) = (1, 2)$.
""")
with st.expander("Solution"):
    st.latex(r"\frac{\partial z}{\partial x} = 6xy")
    st.latex(r"\left.\frac{\partial z}{\partial x}\right|_{(x, y) = (1, 2)} = 6 \times 1 \times 2 = 12")

# --- Problem 2 ---
st.markdown(r"""
---
**Problem 2**  
Surface: $z = x^2 e^y + y^2$  
Find $\frac{\partial z}{\partial y}$ and evaluate at $(x, y) = (2, 0)$.
""")
with st.expander("Solution"):
    st.latex(r"\frac{\partial z}{\partial y} = x^2 e^y + 2y")
    st.latex(r"\left.\frac{\partial z}{\partial y}\right|_{(x, y) = (2, 0)} = 2^2 \cdot e^0 + 2 \cdot 0 = 4 \cdot 1 + 0 = 4")

# --- Problem 3 ---
st.markdown(r"""
---
**Problem 3**  
Surface: $z = \sin(xy)$  
Find $\frac{\partial z}{\partial x}$ and evaluate at $(x, y) = (0, 1)$.
""")
with st.expander("Solution"):
    st.latex(r"\frac{\partial z}{\partial x} = y \cos(xy)")
    st.latex(r"\left.\frac{\partial z}{\partial x}\right|_{(x, y) = (0, 1)} = 1 \cdot \cos(0 \cdot 1) = 1 \cdot 1 = 1")

# --- Problem 4 ---
st.markdown(r"""
---
**Problem 4**  
Surface: $z = \ln(x^2 + y^2 + 1)$  
Find $\frac{\partial z}{\partial y}$ and evaluate at $(x, y) = (1, 2)$.
""")
with st.expander("Solution"):
    st.latex(r"\frac{\partial z}{\partial y} = \frac{2y}{x^2 + y^2 + 1}")
    st.latex(r"\left.\frac{\partial z}{\partial y}\right|_{(x, y) = (1, 2)} = \frac{2 \times 2}{1^2 + 2^2 + 1} = \frac{4}{1 + 4 + 1} = \frac{4}{6} = \frac{2}{3}")

# --- Problem 5 ---
st.markdown(r"""
---
**Problem 5**  
Surface: $z = e^{-x^2 - y^2}$  
Find $\frac{\partial z}{\partial x}$ and evaluate at $(x, y) = (1, 0)$.
""")
with st.expander("Solution"):
    st.latex(r"\frac{\partial z}{\partial x} = -2x e^{-x^2 - y^2}")
    st.latex(r"\left.\frac{\partial z}{\partial x}\right|_{(x, y) = (1, 0)} = -2 \cdot 1 \cdot e^{-1^2 - 0^2} = -2e^{-1} \approx -0.7358")

