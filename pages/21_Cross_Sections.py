import streamlit as st
import numpy as np
import plotly.graph_objs as go

st.title("Cross-Sections (Traces)")

# =========================
# Example 1: Paraboloid Surface
# =========================
st.subheader("Example 1")

st.markdown(
    r"""
    This interactive plot lets you explore **cross-sections (traces)** of the surface $z = x^2 + y^2$.
    - Choose to fix either $x$ or $y$.
    - Select the value at which to fix the variable.
    - The corresponding cross-section curve will be drawn on the surface.
    """
)

# Surface grid
x = np.linspace(-10, 10, 60)
y = np.linspace(-10, 10, 60)
X, Y = np.meshgrid(x, y)
Z = X**2 + Y**2

# Sidebar controls
fix_var = st.radio("Which variable do you want to fix?", ("x", "y"), horizontal=True)
if fix_var == "x":
    fixed_val = st.slider("Select the value of x to fix", float(x[0]), float(x[-1]), 0.0, 0.1)
else:
    fixed_val = st.slider("Select the value of y to fix", float(y[0]), float(y[-1]), 0.0, 0.1)

# Prepare cross-section curve
if fix_var == "x":
    y_cross = np.linspace(-10, 10, 200)
    x_cross = np.full_like(y_cross, fixed_val)
    z_cross = fixed_val**2 + y_cross**2
else:
    x_cross = np.linspace(-10, 10, 200)
    y_cross = np.full_like(x_cross, fixed_val)
    z_cross = x_cross**2 + fixed_val**2

# Add checkboxes to show/hide the surface and the vertical plane
show_surface = st.checkbox("Show surface", value=True)
show_plane = st.checkbox("Show vertical plane", value=True)

# Surface plot (conditionally included)
surface = go.Surface(x=x, y=y, z=Z, colorscale="Viridis", opacity=0.7, showscale=False)

# Cross-section trace
cross_section = go.Scatter3d(
    x=x_cross,
    y=y_cross,
    z=z_cross,
    mode="lines",
    line=dict(color="red", width=6),
    name="Cross-section"
)

# Draw axes
axes = [
    go.Scatter3d(x=[-10, 10], y=[0, 0], z=[0, 0], mode='lines', line=dict(color='black', width=5), name='X-axis'),
    go.Scatter3d(x=[0, 0], y=[-10, 10], z=[0, 0], mode='lines', line=dict(color='black', width=5), name='Y-axis'),
    go.Scatter3d(x=[0, 0], y=[0, 0], z=[0, 200], mode='lines', line=dict(color='black', width=5), name='Z-axis')
]

# Add the xy-plane (z=0)
plane_x = [-10, 10, 10, -10]
plane_y = [-10, -10, 10, 10]
plane_z = [0, 0, 0, 0]
xy_plane = go.Mesh3d(
    x=plane_x,
    y=plane_y,
    z=plane_z,
    color='lightblue',
    opacity=0.3,
    showscale=False,
    name='XY Plane'
)

# Add a lightly colored vertical plane at the fixed value (as a Surface)
if show_plane:
    if fix_var == "x":
        # Plane x = fixed_val, spanning y in [-10, 10], z in [0, 200]
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
        # Plane y = fixed_val, spanning x in [-10, 10], z in [0, 200]
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
data = axes + [cross_section, xy_plane]
if show_surface:
    data.insert(3, surface)  # Insert surface before xy_plane for better layering
if show_plane and vertical_plane is not None:
    data.append(vertical_plane)

# Layout: remove cube, keep axes, no grid, no tick labels
layout = go.Layout(
    scene=dict(
        xaxis=dict(
            range=[-10, 10],
            showgrid=False,
            showbackground=False,
            zeroline=False,
            showline=False,
            showticklabels=False
        ),
        yaxis=dict(
            range=[-10, 10],
            showgrid=False,
            showbackground=False,
            zeroline=False,
            showline=False,
            showticklabels=False
        ),
        zaxis=dict(
            range=[0, 200],
            showgrid=False,
            showbackground=False,
            zeroline=False,
            showline=False,
            showticklabels=False
        ),
    ),
    margin=dict(l=0, r=0, b=0, t=30),
    showlegend=False
)

fig = go.Figure(data=data, layout=layout)

# --- Place 3D and 2D plots side by side ---
col3d, col2d = st.columns(2)

with col3d:
    st.plotly_chart(fig, use_container_width=True)
    st.latex(r"z = x^2 + y^2")

# Prepare 2D trace plot
if fix_var == "x":
    ind_var = y_cross
    ind_label = "y"
    title_2d = f"Trace at x = {fixed_val:.2f}: y vs z"
    plane_color = "rgba(255,165,0,0.10)"  # light orange
    trace_eq = rf"z = {fixed_val:.2f}^2 + y^2"
else:
    ind_var = x_cross
    ind_label = "x"
    title_2d = f"Trace at y = {fixed_val:.2f}: x vs z"
    plane_color = "rgba(255,165,0,0.10)"  # light orange
    trace_eq = rf"z = x^2 + {fixed_val:.2f}^2"

trace2d = go.Scatter(
    x=ind_var,
    y=z_cross,
    mode="lines",
    line=dict(color="red", width=4),
    name="Trace"
)

# Draw explicit axes for 2D plot
axis_x = go.Scatter(
    x=[-10, 10],
    y=[0, 0],
    mode="lines",
    line=dict(color="black", width=3),
    showlegend=False,
    hoverinfo="skip"
)
axis_z = go.Scatter(
    x=[0, 0],
    y=[0, 200],
    mode="lines",
    line=dict(color="black", width=3),
    showlegend=False,
    hoverinfo="skip"
)

layout2d = go.Layout(
    title=title_2d,
    xaxis=dict(
        title=ind_label,
        range=[-10, 10],
        showgrid=False,
        showticklabels=False,
        showline=False,
        zeroline=False
    ),
    yaxis=dict(
        title="z",
        range=[0, 200],
        showgrid=False,
        showticklabels=False,
        showline=False,
        zeroline=False
    ),
    plot_bgcolor=plane_color if show_plane else "white",
    margin=dict(l=40, r=10, b=40, t=40),
    showlegend=False
)

fig2d = go.Figure(data=[axis_x, axis_z, trace2d], layout=layout2d)

with col2d:
    st.plotly_chart(fig2d, use_container_width=True)
    st.latex(trace_eq)

# =========================
# Example 2: Sine Wave Surface
# =========================
st.subheader("Example 2")

st.markdown(
    r"""
    Now explore cross-sections (traces) of the surface $z = \sin(x) + \cos(y)$.
    - Fix either $x$ or $y$ and see the resulting trace.
    """
)

# Surface grid for sine wave
x2 = np.linspace(-10, 10, 60)
y2 = np.linspace(-10, 10, 60)
X2, Y2 = np.meshgrid(x2, y2)
Z2 = np.sin(X2) + np.cos(Y2)

fix_var2 = st.radio("Which variable do you want to fix? (Example 2)", ("x", "y"), horizontal=True, key="ex2_radio")
if fix_var2 == "x":
    fixed_val2 = st.slider("Select the value of x to fix (Example 2)", float(x2[0]), float(x2[-1]), 0.0, 0.1, key="ex2_slider_x")
else:
    fixed_val2 = st.slider("Select the value of y to fix (Example 2)", float(y2[0]), float(y2[-1]), 0.0, 0.1, key="ex2_slider_y")

if fix_var2 == "x":
    y_cross2 = np.linspace(-10, 10, 200)
    x_cross2 = np.full_like(y_cross2, fixed_val2)
    z_cross2 = np.sin(fixed_val2) + np.cos(y_cross2)
else:
    x_cross2 = np.linspace(-10, 10, 200)
    y_cross2 = np.full_like(x_cross2, fixed_val2)
    z_cross2 = np.sin(x_cross2) + np.cos(fixed_val2)

show_surface2 = st.checkbox("Show surface (Example 2)", value=True, key="ex2_surface")
show_plane2 = st.checkbox("Show vertical plane (Example 2)", value=True, key="ex2_plane")

surface2 = go.Surface(x=x2, y=y2, z=Z2, colorscale="Viridis", opacity=0.7, showscale=False)
cross_section2 = go.Scatter3d(
    x=x_cross2,
    y=y_cross2,
    z=z_cross2,
    mode="lines",
    line=dict(color="red", width=6),
    name="Cross-section"
)
axes2 = [
    go.Scatter3d(x=[-10, 10], y=[0, 0], z=[0, 0], mode='lines', line=dict(color='black', width=5)),
    go.Scatter3d(x=[0, 0], y=[-10, 10], z=[0, 0], mode='lines', line=dict(color='black', width=5)),
    go.Scatter3d(x=[0, 0], y=[0, 0], z=[-2, 2], mode='lines', line=dict(color='black', width=5))
]
xy_plane2 = go.Mesh3d(
    x=plane_x,
    y=plane_y,
    z=[0, 0, 0, 0],
    color='lightblue',
    opacity=0.3,
    showscale=False
)
if show_plane2:
    if fix_var2 == "x":
        y_plane2 = np.linspace(-10, 10, 2)
        z_plane2 = np.linspace(-2, 2, 2)
        Y_plane2, Z_plane2 = np.meshgrid(y_plane2, z_plane2)
        X_plane2 = np.full_like(Y_plane2, fixed_val2)
        vertical_plane2 = go.Surface(
            x=X_plane2,
            y=Y_plane2,
            z=Z_plane2,
            showscale=False,
            opacity=0.25,
            colorscale=[[0, 'orange'], [1, 'orange']],
            hoverinfo='skip'
        )
    else:
        x_plane2 = np.linspace(-10, 10, 2)
        z_plane2 = np.linspace(-2, 2, 2)
        X_plane2, Z_plane2 = np.meshgrid(x_plane2, z_plane2)
        Y_plane2 = np.full_like(X_plane2, fixed_val2)
        vertical_plane2 = go.Surface(
            x=X_plane2,
            y=Y_plane2,
            z=Z_plane2,
            showscale=False,
            opacity=0.25,
            colorscale=[[0, 'orange'], [1, 'orange']],
            hoverinfo='skip'
        )
else:
    vertical_plane2 = None

data2 = axes2 + [cross_section2, xy_plane2]
if show_surface2:
    data2.insert(3, surface2)
if show_plane2 and vertical_plane2 is not None:
    data2.append(vertical_plane2)

layout2 = go.Layout(
    scene=dict(
        xaxis=dict(range=[-10, 10], showgrid=False, showbackground=False, zeroline=False, showline=False, showticklabels=False),
        yaxis=dict(range=[-10, 10], showgrid=False, showbackground=False, zeroline=False, showline=False, showticklabels=False),
        zaxis=dict(range=[-2, 2], showgrid=False, showbackground=False, zeroline=False, showline=False, showticklabels=False),
    ),
    margin=dict(l=0, r=0, b=0, t=30),
    showlegend=False
)
fig3d_2 = go.Figure(data=data2, layout=layout2)

col3d2, col2d2 = st.columns(2)
with col3d2:
    st.plotly_chart(fig3d_2, use_container_width=True)
    st.latex(r"z = \sin(x) + \cos(y)")

if fix_var2 == "x":
    ind_var2 = y_cross2
    ind_label2 = "y"
    title_2d2 = f"Trace at x = {fixed_val2:.2f}: y vs z"
    plane_color2 = "rgba(255,165,0,0.10)"
    trace_eq2 = rf"z = \sin({fixed_val2:.2f}) + \cos(y)"
else:
    ind_var2 = x_cross2
    ind_label2 = "x"
    title_2d2 = f"Trace at y = {fixed_val2:.2f}: x vs z"
    plane_color2 = "rgba(255,165,0,0.10)"
    trace_eq2 = rf"z = \sin(x) + \cos({fixed_val2:.2f})"

trace2d_2 = go.Scatter(
    x=ind_var2,
    y=z_cross2,
    mode="lines",
    line=dict(color="red", width=4)
)
axis_x2 = go.Scatter(
    x=[-10, 10],
    y=[0, 0],
    mode="lines",
    line=dict(color="black", width=3),
    showlegend=False,
    hoverinfo="skip"
)
axis_z2 = go.Scatter(
    x=[0, 0],
    y=[-2, 2],
    mode="lines",
    line=dict(color="black", width=3),
    showlegend=False,
    hoverinfo="skip"
)
layout2d_2 = go.Layout(
    title=title_2d2,
    xaxis=dict(title=ind_label2, range=[-10, 10], showgrid=False, showticklabels=False, showline=False, zeroline=False),
    yaxis=dict(title="z", range=[-2, 2], showgrid=False, showticklabels=False, showline=False, zeroline=False),
    plot_bgcolor=plane_color2 if show_plane2 else "white",
    margin=dict(l=40, r=10, b=40, t=40),
    showlegend=False
)
fig2d_2 = go.Figure(data=[axis_x2, axis_z2, trace2d_2], layout=layout2d_2)
with col2d2:
    st.plotly_chart(fig2d_2, use_container_width=True)
    st.latex(trace_eq2)

# =========================
# Example 3: Gaussian Bump Surface (with larger std dev)
# =========================
st.subheader("Example 3")

st.markdown(
    r"""
    Finally, explore cross-sections (traces) of the surface $z = e^{-\frac{x^2 + y^2}{2\sigma^2}}$ with $\sigma = 3$.
    - Fix either $x$ or $y$ and see the resulting trace.
    """
)

sigma = 3  # Larger standard deviation
x3 = np.linspace(-10, 10, 60)
y3 = np.linspace(-10, 10, 60)
X3, Y3 = np.meshgrid(x3, y3)
Z3 = np.exp(-(X3**2 + Y3**2) / (2 * sigma**2))

fix_var3 = st.radio("Which variable do you want to fix? (Example 3)", ("x", "y"), horizontal=True, key="ex3_radio")
if fix_var3 == "x":
    fixed_val3 = st.slider("Select the value of x to fix (Example 3)", float(x3[0]), float(x3[-1]), 0.0, 0.1, key="ex3_slider_x")
else:
    fixed_val3 = st.slider("Select the value of y to fix (Example 3)", float(y3[0]), float(y3[-1]), 0.0, 0.1, key="ex3_slider_y")

if fix_var3 == "x":
    y_cross3 = np.linspace(-10, 10, 200)
    x_cross3 = np.full_like(y_cross3, fixed_val3)
    z_cross3 = np.exp(-(fixed_val3**2 + y_cross3**2) / (2 * sigma**2))
else:
    x_cross3 = np.linspace(-10, 10, 200)
    y_cross3 = np.full_like(x_cross3, fixed_val3)
    z_cross3 = np.exp(-(x_cross3**2 + fixed_val3**2) / (2 * sigma**2))

show_surface3 = st.checkbox("Show surface (Example 3)", value=True, key="ex3_surface")
show_plane3 = st.checkbox("Show vertical plane (Example 3)", value=True, key="ex3_plane")

surface3 = go.Surface(x=x3, y=y3, z=Z3, colorscale="Viridis", opacity=0.7, showscale=False)
cross_section3 = go.Scatter3d(
    x=x_cross3,
    y=y_cross3,
    z=z_cross3,
    mode="lines",
    line=dict(color="red", width=6),
    name="Cross-section"
)
axes3 = [
    go.Scatter3d(x=[-10, 10], y=[0, 0], z=[0, 0], mode='lines', line=dict(color='black', width=5)),
    go.Scatter3d(x=[0, 0], y=[-10, 10], z=[0, 0], mode='lines', line=dict(color='black', width=5)),
    go.Scatter3d(x=[0, 0], y=[0, 0], z=[0, 1], mode='lines', line=dict(color='black', width=5))
]
xy_plane3 = go.Mesh3d(
    x=plane_x,
    y=plane_y,
    z=[0, 0, 0, 0],
    color='lightblue',
    opacity=0.3,
    showscale=False
)
if show_plane3:
    if fix_var3 == "x":
        y_plane3 = np.linspace(-10, 10, 2)
        z_plane3 = np.linspace(0, 1, 2)
        Y_plane3, Z_plane3 = np.meshgrid(y_plane3, z_plane3)
        X_plane3 = np.full_like(Y_plane3, fixed_val3)
        vertical_plane3 = go.Surface(
            x=X_plane3,
            y=Y_plane3,
            z=Z_plane3,
            showscale=False,
            opacity=0.25,
            colorscale=[[0, 'orange'], [1, 'orange']],
            hoverinfo='skip'
        )
    else:
        x_plane3 = np.linspace(-10, 10, 2)
        z_plane3 = np.linspace(0, 1, 2)
        X_plane3, Z_plane3 = np.meshgrid(x_plane3, z_plane3)
        Y_plane3 = np.full_like(X_plane3, fixed_val3)
        vertical_plane3 = go.Surface(
            x=X_plane3,
            y=Y_plane3,
            z=Z_plane3,
            showscale=False,
            opacity=0.25,
            colorscale=[[0, 'orange'], [1, 'orange']],
            hoverinfo='skip'
        )
else:
    vertical_plane3 = None

data3 = axes3 + [cross_section3, xy_plane3]
if show_surface3:
    data3.insert(3, surface3)
if show_plane3 and vertical_plane3 is not None:
    data3.append(vertical_plane3)

layout3 = go.Layout(
    scene=dict(
        xaxis=dict(range=[-10, 10], showgrid=False, showbackground=False, zeroline=False, showline=False, showticklabels=False),
        yaxis=dict(range=[-10, 10], showgrid=False, showbackground=False, zeroline=False, showline=False, showticklabels=False),
        zaxis=dict(range=[0, 1], showgrid=False, showbackground=False, zeroline=False, showline=False, showticklabels=False),
    ),
    margin=dict(l=0, r=0, b=0, t=30),
    showlegend=False
)
fig3d_3 = go.Figure(data=data3, layout=layout3)

col3d3, col2d3 = st.columns(2)
with col3d3:
    st.plotly_chart(fig3d_3, use_container_width=True)
    st.latex(r"z = e^{-\frac{x^2 + y^2}{2 \cdot 3^2}}")

if fix_var3 == "x":
    ind_var3 = y_cross3
    ind_label3 = "y"
    title_2d3 = f"Trace at x = {fixed_val3:.2f}: y vs z"
    plane_color3 = "rgba(255,165,0,0.10)"
    trace_eq3 = rf"z = e^{{-\frac{{{fixed_val3:.2f}^2 + y^2}}{{2 \cdot 3^2}}}}"
else:
    ind_var3 = x_cross3
    ind_label3 = "x"
    title_2d3 = f"Trace at y = {fixed_val3:.2f}: x vs z"
    plane_color3 = "rgba(255,165,0,0.10)"
    trace_eq3 = rf"z = e^{{-\frac{{x^2 + {fixed_val3:.2f}^2}}{{2 \cdot 3^2}}}}"

trace2d_3 = go.Scatter(
    x=ind_var3,
    y=z_cross3,
    mode="lines",
    line=dict(color="red", width=4)
)
axis_x3 = go.Scatter(
    x=[-10, 10],
    y=[0, 0],
    mode="lines",
    line=dict(color="black", width=3),
    showlegend=False,
    hoverinfo="skip"
)
axis_z3 = go.Scatter(
    x=[0, 0],
    y=[0, 1],
    mode="lines",
    line=dict(color="black", width=3),
    showlegend=False,
    hoverinfo="skip"
)
layout2d_3 = go.Layout(
    title=title_2d3,
    xaxis=dict(title=ind_label3, range=[-10, 10], showgrid=False, showticklabels=False, showline=False, zeroline=False),
    yaxis=dict(title="z", range=[0, 1], showgrid=False, showticklabels=False, showline=False, zeroline=False),
    plot_bgcolor=plane_color3 if show_plane3 else "white",
    margin=dict(l=40, r=10, b=40, t=40),
    showlegend=False
)
fig2d_3 = go.Figure(data=[axis_x3, axis_z3, trace2d_3], layout=layout2d_3)
with col2d3:
    st.plotly_chart(fig2d_3, use_container_width=True)
    st.latex(trace_eq3)

# =========================
# Homework: Find the Equation of the Trace
# =========================
st.markdown("---")
st.header("Homework")

st.markdown("""
For each problem below, you are given the equation of a surface and asked to fix one of the independent variables ($x$ or $y$).  
**Write the equation of the resulting 2D trace.**  
Show your work, then check the solution.
""")

# --- Problem 1 ---
st.markdown("""
**Problem 1**  
Surface: $z = 2x^2 + 3y^2 - 4x + 5y - 7$  
Fix $x = 1$
""")
with st.expander("Solution"):
    st.markdown(r"""
$z = 2(1)^2 + 3y^2 - 4(1) + 5y - 7 = 2 + 3y^2 - 4 + 5y - 7 = 3y^2 + 5y - 9$

So, the trace is:  
$\boxed{z = 3y^2 + 5y - 9}$
""")

# --- Problem 2 ---
st.markdown("""
---
**Problem 2**  
Surface: $z = \sin(x) + \cos(y)$  
Fix $y = 0$
""")
with st.expander("Solution"):
    st.markdown(r"""
$z = \sin(x) + \cos(0) = \sin(x) + 1$

So, the trace is:  
$\boxed{z = \sin(x) + 1}$
""")

# --- Problem 3 ---
st.markdown("""
---
**Problem 3**  
Surface: $z = e^{-x^2 - y^2}$  
Fix $x = 2$
""")
with st.expander("Solution"):
    st.markdown(r"""
$z = e^{-(2)^2 - y^2} = e^{-4 - y^2}$

So, the trace is:  
$\boxed{z = e^{-4 - y^2}}$
""")

# --- Problem 4 ---
st.markdown("""
---
**Problem 4**  
Surface: $z = x^2 y + y^2$  
Fix $y = -3$
""")
with st.expander("Solution"):
    st.markdown(r"""
$z = x^2(-3) + (-3)^2 = -3x^2 + 9$

So, the trace is:  
$\boxed{z = -3x^2 + 9}$
""")

# --- Problem 5 ---
st.markdown("""
---
**Problem 5**  
Surface: $z = \ln(x^2 + y^2 + 1)$  
Fix $x = 0$
""")
with st.expander("Solution"):
    st.markdown(r"""
$z = \ln(0^2 + y^2 + 1) = \ln(y^2 + 1)$

So, the trace is:  
$\boxed{z = \ln(y^2 + 1)}$
""")

