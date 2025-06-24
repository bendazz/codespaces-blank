import streamlit as st

st.title("Equation of a Line: Point-Slope Form")

st.markdown("""
To find the equation of a line given a point and the slope, you can use the **point-slope form** of a linear equation:

**Point-Slope Form:**
```
y - y₁ = m(x - x₁)
```
- **m** is the slope of the line.
- **(x₁, y₁)** is a point on the line.

**Steps:**
1. Identify the slope (**m**) and the given point (**(x₁, y₁)**).
2. Substitute these values into the point-slope formula.
3. Simplify the equation if needed.

**Example:**
If the slope is 2 and the point is (3, 4):

```
y - 4 = 2(x - 3)
```
Or, simplified to slope-intercept form (**y = mx + b**):

```
y = 2x - 2
```
""")

st.header("Practice Exercises")

with st.expander("Exercise 1"):
    st.markdown("""
    **Given:** Slope = 5, Point = (1, -2)

    **Solution:**
    ```
    y - (-2) = 5(x - 1)
    y + 2 = 5(x - 1)
    y + 2 = 5x - 5
    y = 5x - 7
    ```
    """)

with st.expander("Exercise 2"):
    st.markdown("""
    **Given:** Slope = -3, Point = (0, 4)

    **Solution:**
    ```
    y - 4 = -3(x - 0)
    y - 4 = -3x
    y = -3x + 4
    ```
    """)

with st.expander("Exercise 3"):
    st.markdown("""
    **Given:** Slope = 1/2, Point = (-2, 3)

    **Solution:**
    ```
    y - 3 = (1/2)(x + 2)
    y - 3 = (1/2)x + 1
    y = (1/2)x + 4
    ```
    """)