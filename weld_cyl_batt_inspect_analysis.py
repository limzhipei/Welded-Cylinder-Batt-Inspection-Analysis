#Import Libraries
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import streamlit as st

st.set_page_config(
     page_title="Welded Cylinder Battery Inspection Analysis Tool",
     page_icon=":shark:",
     layout="centered",
     initial_sidebar_state="expanded",
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
cam1_selected = False
cam2_selected = False

with st.sidebar:
     samples = int(st.number_input('Insert number of samples'))
     st.write('The current number of samples is ', samples)
     selector = st.radio('Which Camera?:', ["Cam 1", "Cam 2"])
     if selector == "Cam 1":
          cam2_selected = False
          cam1_selected = True
          st.write('Cam 1 selected!')
     elif selector == "Cam 2":
          cam1_selected = False
          cam2_selected = True
          st.write('Cam 2 selected!')
          
     uploaded_file = st.file_uploader("Upload your data")
     if uploaded_file is not None:
         # Can be used wherever a "file-like" object is accepted:
         df = pd.read_csv(uploaded_file)
         st.write('File uploaded!')

#Input No of Samples
samples_array = np.arange(1, samples+1)

if (samples == 0 or selector is None or uploaded_file is None):
     st.header('Please input number of samples, select camera and upload a file to start!')

else:
     if cam1_selected == True:
          #Import Cam 1 Data
          cam1_df = df

          #Cam 1 Data Sorting & Pre-processing
          cam1_df = cam1_df.sort_index(axis=1, ascending=True)
          cam1_df = cam1_df.sort_values(by=['Count'] , ascending=True)

          cam1_df_negative = cam1_df.loc[((cam1_df['Count'] % 10) < 6) & ((cam1_df['Count'] % 10) > 0)]
          cam1_df_negative = cam1_df_negative.drop(cam1_df_negative.iloc[:, 1:57],axis = 1)
          cam1_df_negative = cam1_df_negative.drop(cam1_df_negative.iloc[:,70:],axis = 1)

          cam1_df_positive = cam1_df.loc[((cam1_df['Count'] % 10) > 5) | ((cam1_df['Count'] % 10) == 0)]
          cam1_df_positive = cam1_df_positive.drop(cam1_df_positive.iloc[:, 57:],axis = 1)

          #Add Battery No
          cam1_df_negative['Battery No.'] = np.resize(samples_array, cam1_df_negative.shape[0])
          cam1_df_positive['Battery No.'] = np.resize(samples_array, cam1_df_positive.shape[0])
          
          st.write('This is cam 1 negative dataframe')
          st.dataframe(cam1_df_negative)
          
          st.write('This is cam 1 positive dataframe')
          st.dataframe(cam1_df_positive)
          
          #Scatter Plot Selection Negative
          cam1_neg_col = cam1_df_negative.columns.values.tolist()[1:-1]
          option1 = st.selectbox('Choose Y Axis', cam1_neg_col)
          ul1 = int(st.number_input('Upper Limit 1'))
          ll1 = int(st.number_input('Lower Limit 1'))
          
          #Scatter Plots Negative
          st.write('This is cam 1 negative scatter plots')
          x1 = cam1_df_negative['Battery No.']
          
          #Display Plot
          st.write(option1)
          y1 = cam1_df_negative[option1]
          
          plot1 = px.scatter(cam1_df_negative, x=x1, y=y1)
          plot1.add_hline(y=ul1, line_width=3, line_color="red")
          plot1.add_hline(y=ll1, line_width=3, line_color="red")
          st.plotly_chart(plot1, use_container_width=True)
          
          #Scatter Plot Selection Positive
          cam1_pos_col = cam1_df_positive.columns.values.tolist()[1:-1]
          option2 = st.selectbox('Choose Y Axis', cam1_pos_col)
          ul2 = int(st.number_input('Upper Limit 2'))
          ll2 = int(st.number_input('Lower Limit 2'))
          
          #Scatter Plots Positive
          st.write('This is cam 1 positive scatter plots')
          x2 = cam1_df_positive['Battery No.']
          
          #Display Plot
          st.write(option2)
          y2 = cam1_df_positive[option2]
          
          plot2 = px.scatter(cam1_df_positive, x=x2, y=y2)
          plot2.add_hline(y=ul2, line_width=3, line_color="red")
          plot2.add_hline(y=ll2, line_width=3, line_color="red")
          st.plotly_chart(plot2, use_container_width=True)
          
     elif cam2_selected == True:
          #Import Cam 2 Data
          cam2_df = df

          #Cam 2 Data Sorting & Pre-processing
          cam2_df = cam2_df.sort_index(axis=1, ascending=True)
          cam2_df = cam2_df.sort_values(by=['Count'] , ascending=True)

          cam2_df_negative = cam2_df.loc[((cam2_df['Count'] % 10) < 6) & ((cam2_df['Count'] % 10) > 0)]
          cam2_df_negative = cam2_df_negative.drop(cam2_df_negative.iloc[:, 1:183],axis = 1)

          cam2_df_positive = cam2_df.loc[((cam2_df['Count'] % 10) > 5) | ((cam2_df['Count'] % 10) == 0)]
          cam2_df_positive = cam2_df_positive.drop(cam2_df_positive.iloc[:, 1:127],axis = 1)
          cam2_df_positive = cam2_df_positive.drop(cam2_df_positive.iloc[:, 56:],axis = 1)

          #Add Battery No.
          cam2_df_negative['Battery No.'] = np.resize(samples_array, cam2_df_negative.shape[0])
          cam2_df_positive['Battery No.'] = np.resize(samples_array, cam2_df_positive.shape[0])

          st.write('This is cam 2 negative dataframe')
          st.dataframe(cam2_df_negative)
          
          st.write('This is cam 2 positive dataframe')
          st.dataframe(cam2_df_positive)

          #Scatter Plot Selection Negative
          cam2_neg_col = cam2_df_negative.columns.values.tolist()[1:-1]
          option3 = st.selectbox('Choose Y Axis', cam2_neg_col)
          ul3 = int(st.number_input('Upper Limit 3'))
          ll3 = int(st.number_input('Lower Limit 3'))
          
          #Scatter Plots Negative
          st.write('This is cam 2 negative scatter plots')
          x3 = cam2_df_negative['Battery No.']
          
          #Display Plot
          st.write(option3)
          y3 = cam2_df_negative[option3]
          
          plot3 = px.scatter(cam2_df_negative, x=x3, y=y3)
          plot3.add_hline(y=ul3, line_width=3, line_color="red")
          plot3.add_hline(y=ll3, line_width=3, line_color="red")
          st.plotly_chart(plot3, use_container_width=True)
          
          #Scatter Plot Selection Positive
          cam2_pos_col = cam2_df_positive.columns.values.tolist()[1:-1]
          option4 = st.selectbox('Choose Y Axis', cam2_pos_col)
          ul4 = int(st.number_input('Upper Limit 4'))
          ll4 = int(st.number_input('Lower Limit 4'))
          
          #Scatter Plots Positive
          st.write('This is cam 2 positive scatter plots')
          x4 = cam2_df_positive['Battery No.']
          
          #Display Plot
          st.write(option4)
          y4 = cam2_df_positive[option4]
          
          plot4 = px.scatter(cam2_df_positive, x=x4, y=y4)
          plot4.add_hline(y=ul4, line_width=3, line_color="red")
          plot4.add_hline(y=ll4, line_width=3, line_color="red")
          st.plotly_chart(plot4, use_container_width=True)



















