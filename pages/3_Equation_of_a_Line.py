import streamlit as st

st.title("Finding the Equation of a Line Given Two Points")

st.markdown("""
Given two points on a line, **(x₁, y₁)** and **(x₂, y₂)**, you can find the equation of the line in the form **y = mx + b**.

### Steps:

1. **Find the slope (m):**

   The slope is calculated as:
   $$
   m = \\frac{y_2 - y_1}{x_2 - x_1}
   $$

2. **Find the y-intercept (b):**

   Substitute one of the points and the slope into the equation:
   $$
   y_1 = m x_1 + b
   $$
   Solve for **b**:
   $$
   b = y_1 - m x_1
   $$

3. **Write the equation:**

   Substitute the values of **m** and **b** into:
   $$
   y = m x + b
   $$

---

**Example:**  
Given points (2, 3) and (4, 7):

- Slope:  
  $$
  m = \\frac{7 - 3}{4 - 2} = \\frac{4}{2} = 2
  $$
- Y-intercept:  
  $$
  b = 3 - 2 \\times 2 = 3 - 4 = -1
  $$
- Equation:  
  $$
  y = 2x - 1
  $$
  """)

st.header("Practice Exercises")

with st.expander("Exercise 1"):
    st.markdown("""
    **Given points:** (1, 2) and (3, 6)

    **Solution:**

    - Slope:  
      $$
      m = \\frac{6 - 2}{3 - 1} = \\frac{4}{2} = 2
      $$
    - Y-intercept:  
      $$
      b = 2 - 2 \\times 1 = 2 - 2 = 0
      $$
    - Equation:  
      $$
      y = 2x
      $$
    """)

with st.expander("Exercise 2"):
    st.markdown("""
    **Given points:** (-2, 5) and (2, -3)

    **Solution:**

    - Slope:  
      $$
      m = \\frac{-3 - 5}{2 - (-2)} = \\frac{-8}{4} = -2
      $$
    - Y-intercept:  
      $$
      b = 5 - (-2) \\times (-2) = 5 - 4 = 1
      $$
    - Equation:  
      $$
      y = -2x + 1
      $$
    """)

with st.expander("Exercise 3"):
    st.markdown("""
    **Given points:** (0, -1) and (4, 7)

    **Solution:**

    - Slope:  
      $$
      m = \\frac{7 - (-1)}{4 - 0} = \\frac{8}{4} = 2
      $$
    - Y-intercept:  
      $$
      b = -1 - 2 \\times 0 = -1
      $$
    - Equation:  
      $$
      y = 2x - 1
      $$
    """)