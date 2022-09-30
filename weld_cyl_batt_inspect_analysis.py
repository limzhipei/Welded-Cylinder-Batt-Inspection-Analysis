from collections import namedtuple
import altair as alt
import math
import pandas as pd
import streamlit as st

st.set_page_config(
     page_title="Welded Cylinder Battery Inspection Analysis Tool",
     page_icon=":fire:",
     layout="centered",
     initial_sidebar_state="collapsed",
     menu_items={
         'Get Help': None,
         'Report a bug': None,
         'About': "# Welded Cylinder Battery Inspection Analysis Tool"
     }
 )

"""
# Welcome to the Analysis Tool
"""

samples = 0

with st.sidebar:
     selector = st.radio('Select one:', ["default", "strange"])
     if selector == "strange":
          start_spiral_pts = 1675
          start_spiral_turns = 25

total_points = st.slider("Number of points in spiral", 1, 2000, start_spiral_pts)
num_turns = st.slider("Number of turns in spiral", 1, 100, start_spiral_turns)

Point = namedtuple('Point', 'x y')
data = []

points_per_turn = total_points / num_turns


for curr_point_num in range(total_points):
    curr_turn, i = divmod(curr_point_num, points_per_turn)
    angle = (curr_turn + 1) * 2 * math.pi * i / points_per_turn
    radius = curr_point_num / total_points
    x = radius * math.cos(angle)
    y = radius * math.sin(angle)
    data.append(Point(x, y))

st.altair_chart(alt.Chart(pd.DataFrame(data), height=500, width=500)
        .mark_circle(color='#0068c9', opacity=0.5)
        .encode(x='x:Q', y='y:Q'))
