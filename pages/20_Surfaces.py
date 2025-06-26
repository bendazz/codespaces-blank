import streamlit as st
import numpy as np
import plotly.graph_objects as go

st.title("Surfaces")

st.subheader("Example 1")

# Create a grid of x and y values
x = np.linspace(-4, 4, 80)
y = np.linspace(-4, 4, 80)
X, Y = np.meshgrid(x, y)
Z = 10 - X**2 - Y**2

# User input for a point
x_pt = st.number_input("Choose x", min_value=-4.0, max_value=4.0, value=1.0, step=0.1)
y_pt = st.number_input("Choose y", min_value=-4.0, max_value=4.0, value=1.0, step=0.1)
z_pt = 10 - x_pt**2 - y_pt**2

fig = go.Figure()

# Surface (solid color)
fig.add_trace(go.Surface(
    x=X, y=Y, z=Z,
    colorscale=[[0, 'royalblue'], [1, 'royalblue']],
    opacity=0.85,
    showscale=False,
    name='Surface'
))

# Draw extended axes
fig.add_trace(go.Scatter3d(x=[-4, 4], y=[0, 0], z=[0, 0], mode='lines', line=dict(color='black', width=5), name='X-axis'))
fig.add_trace(go.Scatter3d(x=[0, 0], y=[-4, 4], z=[0, 0], mode='lines', line=dict(color='black', width=5), name='Y-axis'))
fig.add_trace(go.Scatter3d(x=[0, 0], y=[0, 0], z=[-10, 12], mode='lines', line=dict(color='black', width=5), name='Z-axis'))

# Add the xy-plane (z=0)
plane_x = [-4, 4, 4, -4]
plane_y = [-4, -4, 4, 4]
plane_z = [0, 0, 0, 0]
fig.add_trace(go.Mesh3d(
    x=plane_x,
    y=plane_y,
    z=plane_z,
    color='lightblue',
    opacity=0.3,
    showscale=False,
    name='XY Plane'
))

# Plot the user-selected point on the surface
fig.add_trace(go.Scatter3d(
    x=[x_pt],
    y=[y_pt],
    z=[z_pt],
    mode='markers+text',
    marker=dict(size=7, color='red'),
    text=[f"({x_pt:.2f}, {y_pt:.2f}, {z_pt:.2f})"],
    textposition="top center",
    name='Selected Point'
))

fig.update_layout(
    scene=dict(
        xaxis=dict(
            range=[-4, 4],
            title='x',
            showbackground=False,
            showgrid=False,
            zeroline=False,
            showline=True,
            linecolor='black',
            linewidth=3,
            ticks='outside',
            tickwidth=2,
            tickcolor='black',
        ),
        yaxis=dict(
            range=[-4, 4],
            title='y',
            showbackground=False,
            showgrid=False,
            zeroline=False,
            showline=True,
            linecolor='black',
            linewidth=3,
            ticks='outside',
            tickwidth=2,
            tickcolor='black',
        ),
        zaxis=dict(
            range=[-10, 12],
            title='z',
            showbackground=False,
            showgrid=False,
            zeroline=False,
            showline=True,
            linecolor='black',
            linewidth=3,
            ticks='outside',
            tickwidth=2,
            tickcolor='black',
        ),
    ),
    margin=dict(l=0, r=0, b=0, t=30),
    showlegend=False
)

st.plotly_chart(fig, use_container_width=True)

st.markdown(r"**Equation of the surface:**")
st.latex(r"z = 10 - x^2 - y^2")

st.subheader("Example 2")

# Create grid and surface
X2, Y2 = np.meshgrid(x, y)
Z2 = X2**2 + Y2**2

# User input for a point
x2_pt = st.number_input("Choose x (Example 2)", min_value=-4.0, max_value=4.0, value=1.0, step=0.1, key="x2")
y2_pt = st.number_input("Choose y (Example 2)", min_value=-4.0, max_value=4.0, value=1.0, step=0.1, key="y2")
z2_pt = x2_pt**2 + y2_pt**2

fig2 = go.Figure()
fig2.add_trace(go.Surface(
    x=X2, y=Y2, z=Z2,
    colorscale=[[0, 'seagreen'], [1, 'seagreen']],
    opacity=0.85,
    showscale=False,
    name='Surface'
))
fig2.add_trace(go.Scatter3d(x=[-4, 4], y=[0, 0], z=[0, 0], mode='lines', line=dict(color='black', width=5)))
fig2.add_trace(go.Scatter3d(x=[0, 0], y=[-4, 4], z=[0, 0], mode='lines', line=dict(color='black', width=5)))
fig2.add_trace(go.Scatter3d(x=[0, 0], y=[0, 0], z=[-10, 20], mode='lines', line=dict(color='black', width=5)))
fig2.add_trace(go.Mesh3d(
    x=plane_x, y=plane_y, z=plane_z,
    color='lightblue', opacity=0.3, showscale=False
))
fig2.add_trace(go.Scatter3d(
    x=[x2_pt], y=[y2_pt], z=[z2_pt],
    mode='markers+text',
    marker=dict(size=7, color='red'),
    text=[f"({x2_pt:.2f}, {y2_pt:.2f}, {z2_pt:.2f})"],
    textposition="top center"
))
fig2.update_layout(
    scene=dict(
        xaxis=dict(range=[-4, 4], title='x', showbackground=False, showgrid=False, zeroline=False, showline=True, linecolor='black', linewidth=3, ticks='outside', tickwidth=2, tickcolor='black'),
        yaxis=dict(range=[-4, 4], title='y', showbackground=False, showgrid=False, zeroline=False, showline=True, linecolor='black', linewidth=3, ticks='outside', tickwidth=2, tickcolor='black'),
        zaxis=dict(range=[-10, 20], title='z', showbackground=False, showgrid=False, zeroline=False, showline=True, linecolor='black', linewidth=3, ticks='outside', tickwidth=2, tickcolor='black'),
    ),
    margin=dict(l=0, r=0, b=0, t=30),
    showlegend=False
)
st.plotly_chart(fig2, use_container_width=True)
st.markdown(r"**Equation of the surface:**")
st.latex(r"z = x^2 + y^2")

st.subheader("Example 3")

X3, Y3 = np.meshgrid(x, y)
Z3 = X3**2 - Y3**2

x3_pt = st.number_input("Choose x (Example 3)", min_value=-4.0, max_value=4.0, value=1.0, step=0.1, key="x3")
y3_pt = st.number_input("Choose y (Example 3)", min_value=-4.0, max_value=4.0, value=1.0, step=0.1, key="y3")
z3_pt = x3_pt**2 - y3_pt**2

fig3 = go.Figure()
fig3.add_trace(go.Surface(
    x=X3, y=Y3, z=Z3,
    colorscale=[[0, 'orange'], [1, 'orange']],
    opacity=0.85,
    showscale=False,
    name='Surface'
))
fig3.add_trace(go.Scatter3d(x=[-4, 4], y=[0, 0], z=[0, 0], mode='lines', line=dict(color='black', width=5)))
fig3.add_trace(go.Scatter3d(x=[0, 0], y=[-4, 4], z=[0, 0], mode='lines', line=dict(color='black', width=5)))
fig3.add_trace(go.Scatter3d(x=[0, 0], y=[0, 0], z=[-10, 12], mode='lines', line=dict(color='black', width=5)))
fig3.add_trace(go.Mesh3d(
    x=plane_x, y=plane_y, z=plane_z,
    color='lightblue', opacity=0.3, showscale=False
))
fig3.add_trace(go.Scatter3d(
    x=[x3_pt], y=[y3_pt], z=[z3_pt],
    mode='markers+text',
    marker=dict(size=7, color='red'),
    text=[f"({x3_pt:.2f}, {y3_pt:.2f}, {z3_pt:.2f})"],
    textposition="top center"
))
fig3.update_layout(
    scene=dict(
        xaxis=dict(range=[-4, 4], title='x', showbackground=False, showgrid=False, zeroline=False, showline=True, linecolor='black', linewidth=3, ticks='outside', tickwidth=2, tickcolor='black'),
        yaxis=dict(range=[-4, 4], title='y', showbackground=False, showgrid=False, zeroline=False, showline=True, linecolor='black', linewidth=3, ticks='outside', tickwidth=2, tickcolor='black'),
        zaxis=dict(range=[-10, 12], title='z', showbackground=False, showgrid=False, zeroline=False, showline=True, linecolor='black', linewidth=3, ticks='outside', tickwidth=2, tickcolor='black'),
    ),
    margin=dict(l=0, r=0, b=0, t=30),
    showlegend=False
)
st.plotly_chart(fig3, use_container_width=True)
st.markdown(r"**Equation of the surface:**")
st.latex(r"z = x^2 - y^2")

# Example 4: z = sin(x) * cos(y)
st.subheader(r"Example 4")

X4, Y4 = np.meshgrid(x, y)
Z4 = np.sin(X4) * np.cos(Y4)

x4_pt = st.number_input("Choose x (Example 4)", min_value=-4.0, max_value=4.0, value=1.0, step=0.1, key="x4")
y4_pt = st.number_input("Choose y (Example 4)", min_value=-4.0, max_value=4.0, value=1.0, step=0.1, key="y4")
z4_pt = np.sin(x4_pt) * np.cos(y4_pt)

fig4 = go.Figure()
fig4.add_trace(go.Surface(
    x=X4, y=Y4, z=Z4,
    colorscale=[[0, 'purple'], [1, 'purple']],
    opacity=0.85,
    showscale=False,
    name='Surface'
))
fig4.add_trace(go.Scatter3d(x=[-4, 4], y=[0, 0], z=[0, 0], mode='lines', line=dict(color='black', width=5)))
fig4.add_trace(go.Scatter3d(x=[0, 0], y=[-4, 4], z=[0, 0], mode='lines', line=dict(color='black', width=5)))
fig4.add_trace(go.Scatter3d(x=[0, 0], y=[0, 0], z=[-2, 2], mode='lines', line=dict(color='black', width=5)))
fig4.add_trace(go.Mesh3d(
    x=plane_x, y=plane_y, z=plane_z,
    color='lightblue', opacity=0.3, showscale=False
))
fig4.add_trace(go.Scatter3d(
    x=[x4_pt], y=[y4_pt], z=[z4_pt],
    mode='markers+text',
    marker=dict(size=7, color='red'),
    text=[f"({x4_pt:.2f}, {y4_pt:.2f}, {z4_pt:.2f})"],
    textposition="top center"
))
fig4.update_layout(
    scene=dict(
        xaxis=dict(range=[-4, 4], title='x', showbackground=False, showgrid=False, zeroline=False, showline=True, linecolor='black', linewidth=3, ticks='outside', tickwidth=2, tickcolor='black'),
        yaxis=dict(range=[-4, 4], title='y', showbackground=False, showgrid=False, zeroline=False, showline=True, linecolor='black', linewidth=3, ticks='outside', tickwidth=2, tickcolor='black'),
        zaxis=dict(range=[-2, 2], title='z', showbackground=False, showgrid=False, zeroline=False, showline=True, linecolor='black', linewidth=3, ticks='outside', tickwidth=2, tickcolor='black'),
    ),
    margin=dict(l=0, r=0, b=0, t=30),
    showlegend=False
)
st.plotly_chart(fig4, use_container_width=True)
st.markdown(r"**Equation of the surface:**")
st.latex(r"z = \sin(x)\cos(y)")

# Example 5: z = exp(-x^2 - y^2)
st.subheader(r"Example 5")

X5, Y5 = np.meshgrid(x, y)
Z5 = np.exp(-X5**2 - Y5**2)

x5_pt = st.number_input("Choose x (Example 5)", min_value=-4.0, max_value=4.0, value=1.0, step=0.1, key="x5")
y5_pt = st.number_input("Choose y (Example 5)", min_value=-4.0, max_value=4.0, value=1.0, step=0.1, key="y5")
z5_pt = np.exp(-x5_pt**2 - y5_pt**2)

fig5 = go.Figure()
fig5.add_trace(go.Surface(
    x=X5, y=Y5, z=Z5,
    colorscale=[[0, 'goldenrod'], [1, 'goldenrod']],
    opacity=0.85,
    showscale=False,
    name='Surface'
))
fig5.add_trace(go.Scatter3d(x=[-4, 4], y=[0, 0], z=[0, 0], mode='lines', line=dict(color='black', width=5)))
fig5.add_trace(go.Scatter3d(x=[0, 0], y=[-4, 4], z=[0, 0], mode='lines', line=dict(color='black', width=5)))
fig5.add_trace(go.Scatter3d(x=[0, 0], y=[0, 0], z=[-0.2, 1.2], mode='lines', line=dict(color='black', width=5)))
fig5.add_trace(go.Mesh3d(
    x=plane_x, y=plane_y, z=plane_z,
    color='lightblue', opacity=0.3, showscale=False
))
fig5.add_trace(go.Scatter3d(
    x=[x5_pt], y=[y5_pt], z=[z5_pt],
    mode='markers+text',
    marker=dict(size=7, color='red'),
    text=[f"({x5_pt:.2f}, {y5_pt:.2f}, {z5_pt:.2f})"],
    textposition="top center"
))
fig5.update_layout(
    scene=dict(
        xaxis=dict(range=[-4, 4], title='x', showbackground=False, showgrid=False, zeroline=False, showline=True, linecolor='black', linewidth=3, ticks='outside', tickwidth=2, tickcolor='black'),
        yaxis=dict(range=[-4, 4], title='y', showbackground=False, showgrid=False, zeroline=False, showline=True, linecolor='black', linewidth=3, ticks='outside', tickwidth=2, tickcolor='black'),
        zaxis=dict(range=[-0.2, 1.2], title='z', showbackground=False, showgrid=False, zeroline=False, showline=True, linecolor='black', linewidth=3, ticks='outside', tickwidth=2, tickcolor='black'),
    ),
    margin=dict(l=0, r=0, b=0, t=30),
    showlegend=False
)
st.plotly_chart(fig5, use_container_width=True)
st.markdown(r"**Equation of the surface:**")
st.latex(r"z = e^{-x^2 - y^2}")

# Example 6: z = x * y
st.subheader(r"Example 6")

X6, Y6 = np.meshgrid(x, y)
Z6 = X6 * Y6

x6_pt = st.number_input("Choose x (Example 6)", min_value=-4.0, max_value=4.0, value=1.0, step=0.1, key="x6")
y6_pt = st.number_input("Choose y (Example 6)", min_value=-4.0, max_value=4.0, value=1.0, step=0.1, key="y6")
z6_pt = x6_pt * y6_pt

fig6 = go.Figure()
fig6.add_trace(go.Surface(
    x=X6, y=Y6, z=Z6,
    colorscale=[[0, 'crimson'], [1, 'crimson']],
    opacity=0.85,
    showscale=False,
    name='Surface'
))
fig6.add_trace(go.Scatter3d(x=[-4, 4], y=[0, 0], z=[0, 0], mode='lines', line=dict(color='black', width=5)))
fig6.add_trace(go.Scatter3d(x=[0, 0], y=[-4, 4], z=[0, 0], mode='lines', line=dict(color='black', width=5)))
fig6.add_trace(go.Scatter3d(x=[0, 0], y=[0, 0], z=[-16, 16], mode='lines', line=dict(color='black', width=5)))
fig6.add_trace(go.Mesh3d(
    x=plane_x, y=plane_y, z=plane_z,
    color='lightblue', opacity=0.3, showscale=False
))
fig6.add_trace(go.Scatter3d(
    x=[x6_pt], y=[y6_pt], z=[z6_pt],
    mode='markers+text',
    marker=dict(size=7, color='red'),
    text=[f"({x6_pt:.2f}, {y6_pt:.2f}, {z6_pt:.2f})"],
    textposition="top center"
))
fig6.update_layout(
    scene=dict(
        xaxis=dict(range=[-4, 4], title='x', showbackground=False, showgrid=False, zeroline=False, showline=True, linecolor='black', linewidth=3, ticks='outside', tickwidth=2, tickcolor='black'),
        yaxis=dict(range=[-4, 4], title='y', showbackground=False, showgrid=False, zeroline=False, showline=True, linecolor='black', linewidth=3, ticks='outside', tickwidth=2, tickcolor='black'),
        zaxis=dict(range=[-16, 16], title='z', showbackground=False, showgrid=False, zeroline=False, showline=True, linecolor='black', linewidth=3, ticks='outside', tickwidth=2, tickcolor='black'),
    ),
    margin=dict(l=0, r=0, b=0, t=30),
    showlegend=False
)
st.plotly_chart(fig6, use_container_width=True)
st.markdown(r"**Equation of the surface:**")
st.latex(r"z = x y")

st.markdown("---")
st.header("Homework")

st.markdown("""
For each problem below, you are given the equation of a surface and values for $x$ and $y$.  
**Calculate $z$** at that point.  
Show your work, then check the solution.
""")

# Problem 1
st.markdown("""
**Problem 1**  
Surface: $z = 2x^2 + 3y^2 - 4x + 5y - 7$  
Given: $x = 1$, $y = -2$
""")
with st.expander("Solution"):
    st.latex(r"""
    \begin{align*}
    z &= 2(1)^2 + 3(-2)^2 - 4(1) + 5(-2) - 7 \\
      &= 2(1) + 3(4) - 4 + (-10) - 7 \\
      &= 2 + 12 - 4 - 10 - 7 \\
      &= 14 - 4 - 10 - 7 \\
      &= 10 - 10 - 7 \\
      &= 0 - 7 \\
      &= -7
    \end{align*}
    """)

# Problem 2
st.markdown("""
**Problem 2**  
Surface: $z = \sin(x^2 + y^2) + x y$  
Given: $x = 1$, $y = \pi$
""")
with st.expander("Solution"):
    st.latex(r"""
    \begin{align*}
    z &= \sin(1^2 + \pi^2) + 1 \cdot \pi \\
      &= \sin(1 + \pi^2) + \pi \\
      &\approx \sin(1 + 9.87) + 3.14 \\
      &\approx \sin(10.87) + 3.14 \\
      &\approx -0.999 + 3.14 \\
      &\approx 2.14
    \end{align*}
    """)

# Problem 3
st.markdown("""
**Problem 3**  
Surface: $z = e^{x-y} + \cos(xy)$  
Given: $x = 0$, $y = 2$
""")
with st.expander("Solution"):
    st.latex(r"""
    \begin{align*}
    z &= e^{0-2} + \cos(0 \cdot 2) \\
      &= e^{-2} + \cos(0) \\
      &\approx 0.1353 + 1 \\
      &= 1.1353
    \end{align*}
    """)

# Problem 4
st.markdown("""
**Problem 4**  
Surface: $z = \ln(x^2 + y^2 + 1)$  
Given: $x = 2$, $y = -2$
""")
with st.expander("Solution"):
    st.latex(r"""
    \begin{align*}
    z &= \ln(2^2 + (-2)^2 + 1) \\
      &= \ln(4 + 4 + 1) \\
      &= \ln(9) \\
      &\approx 2.197
    \end{align*}
    """)

# Problem 5
st.markdown("""
**Problem 5**  
Surface: $z = \dfrac{x^2 - y^2}{x^2 + y^2 + 1}$  
Given: $x = 1$, $y = 2$
""")
with st.expander("Solution"):
    st.latex(r"""
    \begin{align*}
    z &= \frac{1^2 - 2^2}{1^2 + 2^2 + 1} \\
      &= \frac{1 - 4}{1 + 4 + 1} \\
      &= \frac{-3}{6} \\
      &= -0.5
    \end{align*}
    """)

# Problem 6
st.markdown("""
**Problem 6**  
Surface: $z = x^3 - 3x y^2$  
Given: $x = 2$, $y = 1$
""")
with st.expander("Solution"):
    st.latex(r"""
    \begin{align*}
    z &= 2^3 - 3 \cdot 2 \cdot 1^2 \\
      &= 8 - 6 \\
      &= 2
    \end{align*}
    """)