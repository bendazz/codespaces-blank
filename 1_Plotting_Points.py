# ---
# title: Plotting Points
# ---

import streamlit as st
import matplotlib.pyplot as plt



st.title("Plotting a Point on a Cartesian Coordinate System")

st.markdown("""
To plot a point on a Cartesian coordinate system:

1. **Identify the coordinates**: A point is represented as (x, y), where `x` is the horizontal position and `y` is the vertical position.
2. **Locate the x-coordinate**: Start at the origin (0, 0). Move horizontally to the right if `x` is positive, or to the left if `x` is negative.
3. **Locate the y-coordinate**: From the x position, move vertically up if `y` is positive, or down if `y` is negative.
4. **Mark the point**: The intersection of these positions is the location of the point (x, y).
""")

# Interactive coordinate selection
x = st.slider("Select the x-coordinate", -5, 5, 3)
y = st.slider("Select the y-coordinate", -5, 5, 2)

st.markdown(f"""
For example, to plot the point ({x}, {y}):
- Move {abs(x)} units {'right' if x >= 0 else 'left'} from the origin.
- Then move {abs(y)} units {'up' if y >= 0 else 'down'}.
- Mark the point at this location.
""")

# Illustration
fig, ax = plt.subplots()
ax.axhline(0, color='black', linewidth=1)
ax.axvline(0, color='black', linewidth=1)
ax.scatter(x, y, color='red', s=100)
ax.text(x, y + 0.2, f"({x}, {y})", fontsize=12, ha='center')
ax.set_xlim(-5, 5)
ax.set_ylim(-5, 5)
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.grid(True, which='both', linestyle='--', linewidth=0.5)
st.pyplot(fig)

st.markdown("""
---
### Try These Point Plotting Exercises

1. Plot the point **(4, -2)**.
2. Plot the point **(-3, 3)**.
3. Plot the point **(0, -5)**.
4. Plot the point **(-4, 0)**.
5. Plot the point **(2, 5)**.

Use the sliders above to select the coordinates for each exercise and see where the point appears on the graph!
""")