import streamlit as st  
import numpy as np
import altair as alt
import pandas as pd

x1 = st.slider('a1', min_value=-10, max_value=10, value=0, step=1)


y1 = st.slider('a2', min_value=-10, max_value=10, value=0, step=1)


x2 = st.slider('b1', min_value=-10, max_value=10, value=0, step=1)


y2 = st.slider('b2', min_value=-10, max_value=10, value=0, step=1)

st.write(f'$\\text{{A}}: ({x1}, {y1})$')
st.write(f'$\\text{{B}}: ({x2}, {y2})$')

horizontal = x2 - x1
vertical = y2 - y1
slope = vertical / horizontal if horizontal != 0 else np.inf  # Handle vertical line case
st.write(f'$\\text{{Slope}} = \\frac{{\\text{{vertical change}}}}{{\\text{{horizontal change}}}} = \\frac{{{vertical}}}{{{horizontal}}} = {slope}$')

# Midpoints for label positions
mid_x2 = (x1 + x2) / 2
mid_y2 = y1

mid_x3 = x2
mid_y3 = (y1 + y2) / 2

# DataFrames for labels
label_df2 = pd.DataFrame({'x': [mid_x2], 'y': [mid_y2], 'label': [f'{horizontal}']})
label_df3 = pd.DataFrame({'x': [mid_x3], 'y': [mid_y3], 'label': [f'{vertical}']})

# Create a DataFrame for the points
df = pd.DataFrame({
    'x': [x1, x2],
    'y': [y1, y2],
    'label': ['A', 'B']
})

df2 = pd.DataFrame({
    'x': [x1, x2],
    'y': [y1, y1]
})

df3 = pd.DataFrame({
    'x': [x2, x2],
    'y': [y1, y2]
})

# Axes data
x_axis = pd.DataFrame({'x': [-10, 10], 'y': [0, 0]})
y_axis = pd.DataFrame({'x': [0, 0], 'y': [-10, 10]})

# Axes lines
x_axis_line = alt.Chart(x_axis).mark_line(color='black').encode(
    x='x',
    y='y'
)
y_axis_line = alt.Chart(y_axis).mark_line(color='black').encode(
    x='x',
    y='y'
)

# Create the Altair scatter plot
chart = alt.Chart(df).mark_circle(size=200,opacity = 1,stroke = 'black'
).encode(
    x=alt.X('x', scale=alt.Scale(domain=[-10, 10])),
    y=alt.Y('y', scale=alt.Scale(domain=[-10, 10])),
    color=alt.Color('label', legend=None),
    tooltip=['label', 'x', 'y']
).properties(
    width=400,
    height=400
)

# Add text labels to the points
text = alt.Chart(df).mark_text(
    align='left',
    dx=10,
    dy=-10
).encode(
    x='x',
    y='y',
    text='label'
)

# Create the line between the points
line = alt.Chart(df).mark_line(color='blue').encode(
    x='x',
    y='y'
)



dotted2 = alt.Chart(df2).mark_line(color='black',strokeDash = [4,4]).encode(
    x='x',
    y='y'
)

dotted3 = alt.Chart(df3).mark_line(color='black',strokeDash = [4,4]).encode(
    x='x',
    y='y'
)

# Add horizontal and vertical labels
label2 = alt.Chart(label_df2).mark_text(
    align='center',
    dy=-10,
    fontSize=14,
    color='black'
).encode(
    x='x',
    y='y',
    text='label'
)
label3 = alt.Chart(label_df3).mark_text(
    align='left',
    dx=10,
    fontSize=14,
    color='black'
).encode(
    x='x',
    y='y',
    text='label'
)

st.altair_chart(dotted3 + dotted2 + label3 + label2 + line + x_axis_line + y_axis_line + chart + text, use_container_width=True)










