import streamlit as st
import plotly.graph_objects as go

st.title("3D Plotting: Interactive Point")

# User input for point coordinates
x_pt = st.number_input("X coordinate", min_value=-5.0, max_value=5.0, value=1.0, step=0.1)
y_pt = st.number_input("Y coordinate", min_value=-5.0, max_value=5.0, value=2.0, step=0.1)
z_pt = st.number_input("Z coordinate", min_value=-5.0, max_value=5.0, value=3.0, step=0.1)

fig = go.Figure()

# Draw positive axes (solid lines) - X and Y reversed
fig.add_trace(go.Scatter3d(x=[0, 0], y=[0, 4], z=[0, 0], mode='lines', line=dict(color='black', width=5), name='X-axis'))  # Now Y-axis
fig.add_trace(go.Scatter3d(x=[0, 4], y=[0, 0], z=[0, 0], mode='lines', line=dict(color='black', width=5), name='Y-axis'))  # Now X-axis
fig.add_trace(go.Scatter3d(x=[0, 0], y=[0, 0], z=[0, 4], mode='lines', line=dict(color='black', width=5), name='Z-axis'))

# Draw negative axes (dotted lines using markers) - X and Y reversed
fig.add_trace(go.Scatter3d(x=[0, 0], y=[0, -4], z=[0, 0], mode='markers+lines',
    line=dict(color='black', width=3, dash='dot'), marker=dict(size=4, color='black'), name='-X-axis'))  # Now -Y-axis
fig.add_trace(go.Scatter3d(x=[0, -4], y=[0, 0], z=[0, 0], mode='markers+lines',
    line=dict(color='black', width=3, dash='dot'), marker=dict(size=4, color='black'), name='-Y-axis'))  # Now -X-axis
fig.add_trace(go.Scatter3d(x=[0, 0], y=[0, 0], z=[0, -4], mode='markers+lines',
    line=dict(color='black', width=3, dash='dot'), marker=dict(size=4, color='black'), name='-Z-axis'))

# Add the xy-plane (z=0) - X and Y reversed
plane_x = [-5, 5, 5, -5]
plane_y = [-5, -5, 5, 5]
plane_z = [0, 0, 0, 0]
fig.add_trace(go.Mesh3d(
    x=plane_y,  # swapped
    y=plane_x,  # swapped
    z=plane_z,
    color='lightblue',
    opacity=0.4,
    showscale=False,
    name='XY Plane'
))

# Plot the user-selected point
fig.add_trace(go.Scatter3d(
    x=[y_pt],  # swapped
    y=[x_pt],  # swapped
    z=[z_pt],
    mode='markers+text',
    marker=dict(size=7, color='red'),
    text=[f"({x_pt:.1f}, {y_pt:.1f}, {z_pt:.1f})"],
    textposition="top center",
    name='Selected Point'
))

fig.update_layout(
    scene=dict(
        xaxis=dict(range=[-5, 5], title='Y'),  # swapped
        yaxis=dict(range=[-5, 5], title='X'),  # swapped
        zaxis=dict(range=[-5, 5], title='Z'),
        aspectmode='cube'
    ),
    showlegend=False,
    margin=dict(l=0, r=0, b=0, t=0)
)

st.plotly_chart(fig, use_container_width=True)

st.markdown("---")
st.header("Homework: Visualizing and Plotting Points in 3D")

st.markdown("""
For each of the following, first **visualize in your mind** where the point will be in the 3D coordinate system.  
Then, use the interactive tool above to plot the point and check your intuition.

1. Plot the point (2, 1, 3).
2. Plot the point (-3, 2, 0).
3. Plot the point (0, -4, 2).
4. Plot the point (1, 1, 1).
5. Plot the point (0, 0, 4).
6. Plot the point (4, 0, 0).
7. Plot the point (0, 3, -2).
8. Plot the point (-2, -2, -2).
9. Plot the point (3, -1, 2).
10. Plot the point (-4, 4, -1).

For each problem, before you plot, ask yourself:
- Which octant/quadrant will the point be in?
- How far is it from the origin along each axis?
- Will it be above, below, left, right, in front, or behind the origin?

Then, enter the coordinates in the interactive tool and see if your mental image matches the plot!
""")