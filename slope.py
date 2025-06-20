import streamlit as st  
import numpy as np
import altair as alt
import pandas as pd

x1 = st.slider('x1', min_value=-10, max_value=10, value=0, step=1)


y1 = st.slider('y1', min_value=-10, max_value=10, value=0, step=1)


x2 = st.slider('x2', min_value=-10, max_value=10, value=0, step=1)


y2 = st.slider('y2', min_value=-10, max_value=10, value=0, step=1)

st.write(f'$\\text{{A}}: ({x1}, {y1})$')
st.write(f'$\\text{{B}}: ({x2}, {y2})$')










