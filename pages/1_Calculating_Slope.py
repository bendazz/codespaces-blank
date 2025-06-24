# ---
# title: Slope of a Line
# ---

import streamlit as st
import matplotlib.pyplot as plt

# Set up the Streamlit page configuration
st.set_page_config(page_title="Slope of a Line", page_icon="📐")


st.set_page_config(page_title="How to Calculate the Slope of a Line")

st.markdown("""
# How to Calculate the Slope of a Line

To calculate the slope of a line, you need two distinct points on the line, typically represented as <code>(x₁, y₁)</code> and <code>(x₂, y₂)</code>. The slope measures how steep the line is and is defined as the ratio of the vertical change (rise) to the horizontal change (run) between these two points.

Mathematically, the slope (often denoted as <code>m</code>) is calculated using the formula:

<p style="text-align: center;">
    <code>m = (y₂ - y₁) / (x₂ - x₁)</code>
</p>

Here, <code>(y₂ - y₁)</code> represents the change in the y-coordinates (vertical direction), and <code>(x₂ - x₁)</code> represents the change in the x-coordinates (horizontal direction). The result tells you how much the y-value increases or decreases for each unit increase in the x-value.

If the slope is positive, the line rises as it moves from left to right. If the slope is negative, the line falls as it moves from left to right. A slope of zero means the line is horizontal, while an undefined slope (when <code>x₂ = x₁</code>) means the line is vertical. 
""", unsafe_allow_html=True)

st.markdown("---")
st.markdown("## Example with Illustration")

# Example points (static)
x1, y1 = 1, 2
x2, y2 = 4, 5

# Calculate slope
m = (y2 - y1) / (x2 - x1)

st.write(f"Let's use the points **A({x1}, {y1})** and **B({x2}, {y2})**.")
st.latex(r"m = \frac{y_2 - y_1}{x_2 - x_1} = \frac{%d - %d}{%d - %d} = %.2f" % (y2, y1, x2, x1, m))

# Plotting static example
fig, ax = plt.subplots()
ax.plot([x1, x2], [y1, y2], marker='o', color='b', label='Line AB')
ax.annotate(f"A({x1},{y1})", (x1, y1), textcoords="offset points", xytext=(-20,10), ha='center')
ax.annotate(f"B({x2},{y2})", (x2, y2), textcoords="offset points", xytext=(20,-10), ha='center')
ax.plot([x1, x2], [y1, y1], 'r--', linewidth=1)
ax.plot([x2, x2], [y1, y2], 'g--', linewidth=1)
ax.text((x1+x2)/2, y1-0.3, "run", color='red', ha='center')
ax.text(x2+0.2, (y1+y2)/2, "rise", color='green', va='center')

ax.set_xlim(0, 5)
ax.set_ylim(1, 6)
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_title("Illustration of Slope Calculation")
ax.grid(True)
st.pyplot(fig)

st.markdown("---")
st.markdown("## Interactive Example")

st.write("Move the sliders to change the coordinates of points A and B:")

col1, col2 = st.columns(2)
with col1:
    x1_int = st.slider("A: x₁", 0, 10, 1)
    y1_int = st.slider("A: y₁", 0, 10, 2)
with col2:
    x2_int = st.slider("B: x₂", 0, 10, 4)
    y2_int = st.slider("B: y₂", 0, 10, 5)

if x2_int == x1_int:
    st.error("x₂ cannot be equal to x₁ (vertical line, slope undefined).")
else:
    m_int = (y2_int - y1_int) / (x2_int - x1_int)
    st.write(f"**A({x1_int}, {y1_int})**, **B({x2_int}, {y2_int})**")
    st.latex(r"m = \frac{y_2 - y_1}{x_2 - x_1} = \frac{%d - %d}{%d - %d} = %.2f" % (y2_int, y1_int, x2_int, x1_int, m_int))

    fig2, ax2 = plt.subplots()
    ax2.plot([x1_int, x2_int], [y1_int, y2_int], marker='o', color='b', label='Line AB')
    ax2.annotate(f"A({x1_int},{y1_int})", (x1_int, y1_int), textcoords="offset points", xytext=(-20,10), ha='center')
    ax2.annotate(f"B({x2_int},{y2_int})", (x2_int, y2_int), textcoords="offset points", xytext=(20,-10), ha='center')
    ax2.plot([x1_int, x2_int], [y1_int, y1_int], 'r--', linewidth=1)
    ax2.plot([x2_int, x2_int], [y1_int, y2_int], 'g--', linewidth=1)
    ax2.text((x1_int+x2_int)/2, y1_int-0.3, "run", color='red', ha='center')
    ax2.text(x2_int+0.2, (y1_int+y2_int)/2, "rise", color='green', va='center')

    ax2.set_xlim(0, 10)
    ax2.set_ylim(0, 10)
    ax2.set_xlabel("x")
    ax2.set_ylabel("y")
    ax2.set_title("Interactive Slope Illustration")
    ax2.grid(True)
    st.pyplot(fig2)

st.markdown("---")
st.markdown("## Practice Exercises")

st.markdown("""
Try solving these exercises to practice calculating the slope of a line:

**1.** What is the slope of the line passing through the points (2, 3) and (5, 9)?

**2.** Find the slope of the line that goes through (0, 0) and (4, 2).

**3.** Calculate the slope for the points (-1, -2) and (3, 6).

**4.** What is the slope of a line passing through (7, 5) and (7, -3)?  
(Hint: What happens when the x-coordinates are the same?)

**5.** If the slope between (a, 4) and (6, 10) is 2, what is the value of a?

**6.** Find the slope of the line through (1, 7) and (4, 7).

**7.** What is the slope of the line passing through (3, -2) and (6, 4)?

**8.** Calculate the slope for the points (0, 5) and (10, 5).

**9.** What is the slope of the line passing through (-3, 2) and (3, -4)?

**10.** If the slope between (x, 1) and (5, 13) is 3, what is the value of x?
""")

st.markdown("---")
st.markdown("## Answers to Practice Exercises")

st.markdown("""
**1.** Slope = (9 - 3) / (5 - 2) = 6 / 3 = **2**

**2.** Slope = (2 - 0) / (4 - 0) = 2 / 4 = **0.5**

**3.** Slope = (6 - (-2)) / (3 - (-1)) = 8 / 4 = **2**

**4.** Slope = (-3 - 5) / (7 - 7) = -8 / 0 = **undefined** (vertical line)

**5.** 2 = (10 - 4) / (6 - a) → 2 = 6 / (6 - a)  
  2(6 - a) = 6  
  12 - 2a = 6  
  -2a = -6  
  a = **3**

**6.** Slope = (7 - 7) / (4 - 1) = 0 / 3 = **0** (horizontal line)

**7.** Slope = (4 - (-2)) / (6 - 3) = 6 / 3 = **2**

**8.** Slope = (5 - 5) / (10 - 0) = 0 / 10 = **0** (horizontal line)

**9.** Slope = (-4 - 2) / (3 - (-3)) = (-6) / 6 = **-1**

**10.** 3 = (13 - 1) / (5 - x) → 3 = 12 / (5 - x)  
  3(5 - x) = 12  
  15 - 3x = 12  
  -3x = 12 - 15 = -3  
  x = **1**
""")