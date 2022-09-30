#Import Libraries
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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
cam1_selected = False
cam2_selected = False

with st.sidebar:
     samples = st.number_input('Insert number of samples')
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
          cam1_df_negative = cam1_df_negative.drop(cam1_df_negative.iloc[:, 1:43],axis = 1)
          cam1_df_negative = cam1_df_negative.drop(cam1_df_negative.iloc[:,55:],axis = 1)

          cam1_df_positive = cam1_df.loc[((cam1_df['Count'] % 10) > 5) | ((cam1_df['Count'] % 10) == 0)]
          cam1_df_positive = cam1_df_positive.drop(cam1_df_positive.iloc[:, 43:],axis = 1)

          #Add Battery No
          cam1_df_negative['Battery No.'] = np.resize(samples_array, cam1_df_negative.shape[0])
          cam1_df_positive['Battery No.'] = np.resize(samples_array, cam1_df_positive.shape[0])
          
          st.write('This is cam 1 negative dataframe')
          st.dataframe(cam1_df_negative)
          
          st.write('This is cam 1 positive dataframe')
          st.dataframe(cam1_df_positive)
          
          #Scatter Plot Selection
          cam1_neg_col = cam1_df_negative.columns.values.tolist()
          option1 = st.selectbox('Choose Y Axis', cam1_neg_col)
          
          #Scatter Plots Negative
          st.write('This is cam 1 negative scatter plots')
          x = cam1_df_negative['Battery No.'].tolist()
          
          #T151: Pattern Match
          st.write(option1)
          y1 = cam1_df_negative.loc[:, [option1]]
          
          fig1 = plt.figure()
          plt.scatter(x, y1, c ="blue")
          st.write(fig1)
          
          #T152: Find Weld
          st.write('T152: Find Weld')
          y2 = cam1_df_negative.iloc[:, [6]]
          
          fig2 = plt.figure()
          plt.scatter(x, y2, c ="blue")
          st.write(fig2)
          
          #T157: Uki 1
          st.write('T157: Uki 1')
          y3 = cam1_df_negative.iloc[:, [8]]
          
          fig3 = plt.figure()
          plt.scatter(x, y3, c ="blue")
          st.write(fig3)
          
          #Scatter Plots Positive
          st.write('This is cam 1 positive scatter plots')
          

     elif cam2_selected == True:
          #Import Cam 2 Data
          cam2_df = pd.read_csv("C:/Users/70N5593/Desktop/20220924 Welded Cylinder Battery Excel Data/Raw data/220924_NG_Cam2.csv")

          #Cam 2 Data Sorting & Pre-processing
          cam2_df = cam2_df.sort_index(axis=1, ascending=True)
          cam2_df = cam2_df.sort_values(by=['Count'] , ascending=True)

          cam2_df_negative = cam2_df.loc[((cam2_df['Count'] % 10) < 6) & ((cam2_df['Count'] % 10) > 0)]
          cam2_df_negative = cam2_df_negative.drop(cam2_df_negative.iloc[:, 1:139],axis = 1)

          cam2_df_positive = cam2_df.loc[((cam2_df['Count'] % 10) > 5) | ((cam2_df['Count'] % 10) == 0)]
          cam2_df_positive = cam2_df_positive.drop(cam2_df_positive.iloc[:, 1:97],axis = 1)
          cam2_df_positive = cam2_df_positive.drop(cam2_df_positive.iloc[:, 43:],axis = 1)

          #Add Battery No.
          cam2_df_negative['Battery No.'] = np.resize(samples_array, cam2_df_negative.shape[0])
          cam2_df_positive['Battery No.'] = np.resize(samples_array, cam2_df_positive.shape[0])

          st.write('This is cam 2 negative dataframe')
          st.dataframe(cam1_df_negative)
          
          st.write('This is cam 2 positive dataframe')
          st.dataframe(cam1_df_positive)





















