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
     selector = st.radio('Which Model?:', ["23N", "2PN"])
     if selector == "23N":
          model_2PN_selected = False
          model_23N_selected = True
          st.write('Model 23N selected!')
     elif selector == "2PN":
          model_23N_selected = False
          model_2PN_selected = True
          st.write('Model 2PN selected!')
          
     uploaded_file = st.file_uploader("Upload your data")
     if uploaded_file is not None:
         # Can be used wherever a "file-like" object is accepted:
         df = pd.read_csv(uploaded_file)
         st.write('File uploaded!')

#Input No of Samples
samples_array = np.arange(1, samples+1)

if (samples == 0 or selector is None or uploaded_file is None):
     with st.container():
          st.subheader('Please input number of samples, select camera, select model and upload a file to start!')
          st.info('Battery number must be consecutive in the test run.')

else:
     #23N Selected
     if model_23N_selected == True:
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
               col1, col2, col3, col4 = st.columns(4)
               
               with col1:
                    option1 = st.selectbox('Choose Y Axis', cam1_neg_col)
                    ul1 = int(st.number_input('Upper Limit 1'))
                    ll1 = int(st.number_input('Lower Limit 1'))

                    #Scatter Plots Negative
                    st.write('This is cam 1 negative scatter plots')
                    x1 = cam1_df_negative['Battery No.']

                    #Display Plot
                    y1 = cam1_df_negative[option1]

                    plot1 = px.scatter(cam1_df_negative, x=x1, y=y1)
                    plot1.add_hline(y=ul1, line_width=3, line_color="red")
                    plot1.add_hline(y=ll1, line_width=3, line_color="red")
                    plot1.add_hrect(y0=ll1, y1=ul1, line_width=0, fillcolor="green", opacity=0.2)
                    st.plotly_chart(plot1, use_container_width=True)
                    
               with col2:
                    option2 = st.selectbox('Choose Y Axis', cam1_neg_col)
                    ul2 = int(st.number_input('Upper Limit 2'))
                    ll2 = int(st.number_input('Lower Limit 2'))

                    #Scatter Plots Negative
                    st.write('This is cam 1 negative scatter plots')
                    x2 = cam1_df_negative['Battery No.']

                    #Display Plot
                    y2 = cam1_df_negative[option2]

                    plot2 = px.scatter(cam1_df_negative, x=x2, y=y2)
                    plot2.add_hline(y=ul2, line_width=3, line_color="red")
                    plot2.add_hline(y=ll2, line_width=3, line_color="red")
                    plot2.add_hrect(y0=ll2, y1=ul2, line_width=0, fillcolor="green", opacity=0.2)
                    st.plotly_chart(plot2, use_container_width=True)
                    
               with col3:
                    option3 = st.selectbox('Choose Y Axis', cam1_neg_col)
                    ul3 = int(st.number_input('Upper Limit 3'))
                    ll3 = int(st.number_input('Lower Limit 3'))

                    #Scatter Plots Negative
                    st.write('This is cam 1 negative scatter plots')
                    x3 = cam1_df_negative['Battery No.']

                    #Display Plot
                    y3 = cam1_df_negative[option3]

                    plot3 = px.scatter(cam1_df_negative, x=x3, y=y3)
                    plot3.add_hline(y=ul3, line_width=3, line_color="red")
                    plot3.add_hline(y=ll3, line_width=3, line_color="red")
                    plot3.add_hrect(y0=ll3, y1=ul3, line_width=0, fillcolor="green", opacity=0.2)
                    st.plotly_chart(plot3, use_container_width=True)
                    
               with col4:
                    option4 = st.selectbox('Choose Y Axis', cam1_neg_col)
                    ul4 = int(st.number_input('Upper Limit 4'))
                    ll4 = int(st.number_input('Lower Limit 4'))

                    #Scatter Plots Negative
                    st.write('This is cam 1 negative scatter plots')
                    x4 = cam1_df_negative['Battery No.']

                    #Display Plot
                    y4 = cam1_df_negative[option4]

                    plot4 = px.scatter(cam1_df_negative, x=x4, y=y4)
                    plot4.add_hline(y=ul4, line_width=3, line_color="red")
                    plot4.add_hline(y=ll4, line_width=3, line_color="red")
                    plot4.add_hrect(y0=ll4, y1=ul4, line_width=0, fillcolor="green", opacity=0.2)
                    st.plotly_chart(plot4, use_container_width=True)
               

               #Scatter Plot Selection Positive
               col5, col6, col7, col8 = st.columns(4)
               cam1_pos_col = cam1_df_positive.columns.values.tolist()[1:-1]
               
               with col5:
                    option5 = st.selectbox('Choose Y Axis', cam1_pos_col)
                    ul5 = int(st.number_input('Upper Limit 5'))
                    ll5 = int(st.number_input('Lower Limit 5'))

                    #Scatter Plots Positive
                    st.write('This is cam 1 positive scatter plots')
                    x5 = cam1_df_positive['Battery No.']

                    #Display Plot
                    y5 = cam1_df_positive[option5]

                    plot5 = px.scatter(cam1_df_positive, x=x5, y=y5)
                    plot5.add_hline(y=ul5, line_width=3, line_color="red")
                    plot5.add_hline(y=ll5, line_width=3, line_color="red")
                    plot5.add_hrect(y0=ll5, y1=ul5, line_width=0, fillcolor="green", opacity=0.2)
                    st.plotly_chart(plot5, use_container_width=True)
                    
               with col6:
                    option6 = st.selectbox('Choose Y Axis', cam1_pos_col)
                    ul6 = int(st.number_input('Upper Limit 6'))
                    ll6 = int(st.number_input('Lower Limit 6'))

                    #Scatter Plots Positive
                    st.write('This is cam 1 positive scatter plots')
                    x6 = cam1_df_positive['Battery No.']

                    #Display Plot
                    y6 = cam1_df_positive[option6]

                    plot6 = px.scatter(cam1_df_positive, x=x6, y=y6)
                    plot6.add_hline(y=ul6, line_width=3, line_color="red")
                    plot6.add_hline(y=ll6, line_width=3, line_color="red")
                    plot6.add_hrect(y0=ll6, y1=ul6, line_width=0, fillcolor="green", opacity=0.2)
                    st.plotly_chart(plot6, use_container_width=True)
                    
               with col7:
                    option7 = st.selectbox('Choose Y Axis', cam1_pos_col)
                    ul7 = int(st.number_input('Upper Limit 7'))
                    ll7 = int(st.number_input('Lower Limit 7'))

                    #Scatter Plots Positive
                    st.write('This is cam 1 positive scatter plots')
                    x7 = cam1_df_positive['Battery No.']

                    #Display Plot
                    y7 = cam1_df_positive[option7]

                    plot7 = px.scatter(cam1_df_positive, x=x7, y=y7)
                    plot7.add_hline(y=ul7, line_width=3, line_color="red")
                    plot7.add_hline(y=ll7, line_width=3, line_color="red")
                    plot7.add_hrect(y0=ll7, y1=ul7, line_width=0, fillcolor="green", opacity=0.2)
                    st.plotly_chart(plot7, use_container_width=True)
                    
               with col8:
                    option8 = st.selectbox('Choose Y Axis', cam1_pos_col)
                    ul8 = int(st.number_input('Upper Limit 8'))
                    ll8 = int(st.number_input('Lower Limit 8'))

                    #Scatter Plots Positive
                    st.write('This is cam 1 positive scatter plots')
                    x8 = cam1_df_positive['Battery No.']

                    #Display Plot
                    y8 = cam1_df_positive[option8]

                    plot8 = px.scatter(cam1_df_positive, x=x8, y=y8)
                    plot8.add_hline(y=ul8, line_width=3, line_color="red")
                    plot8.add_hline(y=ll8, line_width=3, line_color="red")
                    plot8.add_hrect(y0=ll8, y1=ul8, line_width=0, fillcolor="green", opacity=0.2)
                    st.plotly_chart(plot8, use_container_width=True)

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
               col9, col10, col11, col12 = st.columns(4)
               
               with col9:
                    option9 = st.selectbox('Choose Y Axis', cam2_neg_col)
                    ul9 = int(st.number_input('Upper Limit 9'))
                    ll9 = int(st.number_input('Lower Limit 9'))

                    #Scatter Plots Negative
                    st.write('This is cam 2 negative scatter plots')
                    x9 = cam2_df_negative['Battery No.']

                    #Display Plot
                    y9 = cam2_df_negative[option9]

                    plot9 = px.scatter(cam2_df_negative, x=x9, y=y9)
                    plot9.add_hline(y=ul9, line_width=3, line_color="red")
                    plot9.add_hline(y=ll9, line_width=3, line_color="red")
                    plot9.add_hrect(y0=ll9, y1=ul9, line_width=0, fillcolor="green", opacity=0.2)
                    st.plotly_chart(plot9, use_container_width=True)
                    
               with col10:
                    option10 = st.selectbox('Choose Y Axis', cam2_neg_col)
                    ul10 = int(st.number_input('Upper Limit 10'))
                    ll10 = int(st.number_input('Lower Limit 10'))

                    #Scatter Plots Negative
                    st.write('This is cam 2 negative scatter plots')
                    x10 = cam2_df_negative['Battery No.']

                    #Display Plot
                    y10 = cam2_df_negative[option10]

                    plot10 = px.scatter(cam2_df_negative, x=x10, y=y10)
                    plot10.add_hline(y=ul10, line_width=3, line_color="red")
                    plot10.add_hline(y=ll10, line_width=3, line_color="red")
                    plot10.add_hrect(y0=ll10, y1=ul10, line_width=0, fillcolor="green", opacity=0.2)
                    st.plotly_chart(plot10, use_container_width=True)
                    
               with col11:
                    option11 = st.selectbox('Choose Y Axis', cam2_neg_col)
                    ul11 = int(st.number_input('Upper Limit 11'))
                    ll11 = int(st.number_input('Lower Limit 11'))

                    #Scatter Plots Negative
                    st.write('This is cam 2 negative scatter plots')
                    x11 = cam2_df_negative['Battery No.']

                    #Display Plot
                    y11 = cam2_df_negative[option11]

                    plot11 = px.scatter(cam2_df_negative, x=x11, y=y11)
                    plot11.add_hline(y=ul11, line_width=3, line_color="red")
                    plot11.add_hline(y=ll11, line_width=3, line_color="red")
                    plot11.add_hrect(y0=ll11, y1=ul11, line_width=0, fillcolor="green", opacity=0.2)
                    st.plotly_chart(plot11, use_container_width=True)
                    
               with col12:
                    option12 = st.selectbox('Choose Y Axis', cam2_neg_col)
                    ul12 = int(st.number_input('Upper Limit 12'))
                    ll12 = int(st.number_input('Lower Limit 12'))

                    #Scatter Plots Negative
                    st.write('This is cam 2 negative scatter plots')
                    x12 = cam2_df_negative['Battery No.']

                    #Display Plot
                    y12 = cam2_df_negative[option12]

                    plot12 = px.scatter(cam2_df_negative, x=x12, y=y12)
                    plot12.add_hline(y=ul12, line_width=3, line_color="red")
                    plot12.add_hline(y=ll12, line_width=3, line_color="red")
                    plot12.add_hrect(y0=ll12, y1=ul12, line_width=0, fillcolor="green", opacity=0.2)
                    st.plotly_chart(plot12, use_container_width=True)

               #Scatter Plot Selection Positive
               cam2_pos_col = cam2_df_positive.columns.values.tolist()[1:-1]
               col13, col14, col15, col16 = st.columns(4)
               
               with col13:
                    option13 = st.selectbox('Choose Y Axis', cam2_pos_col)
                    ul13 = int(st.number_input('Upper Limit 13'))
                    ll13 = int(st.number_input('Lower Limit 13'))

                    #Scatter Plots Positive
                    st.write('This is cam 2 positive scatter plots')
                    x13 = cam2_df_positive['Battery No.']

                    #Display Plot
                    y13 = cam2_df_positive[option13]

                    plot13 = px.scatter(cam2_df_positive, x=x13, y=y13)
                    plot13.add_hline(y=ul13, line_width=3, line_color="red")
                    plot13.add_hline(y=ll13, line_width=3, line_color="red")
                    plot13.add_hrect(y0=ll13, y1=ul13, line_width=0, fillcolor="green", opacity=0.2)
                    st.plotly_chart(plot13, use_container_width=True)
                    
               with col14:
                    option14 = st.selectbox('Choose Y Axis', cam2_pos_col)
                    ul14 = int(st.number_input('Upper Limit 14'))
                    ll14 = int(st.number_input('Lower Limit 14'))

                    #Scatter Plots Positive
                    st.write('This is cam 2 positive scatter plots')
                    x14 = cam2_df_positive['Battery No.']

                    #Display Plot
                    y14 = cam2_df_positive[option14]

                    plot14 = px.scatter(cam2_df_positive, x=x14, y=y14)
                    plot14.add_hline(y=ul14, line_width=3, line_color="red")
                    plot14.add_hline(y=ll14, line_width=3, line_color="red")
                    plot14.add_hrect(y0=ll14, y1=ul14, line_width=0, fillcolor="green", opacity=0.2)
                    st.plotly_chart(plot14, use_container_width=True)
                    
               with col15:
                    option15 = st.selectbox('Choose Y Axis', cam2_pos_col)
                    ul15 = int(st.number_input('Upper Limit 15'))
                    ll15 = int(st.number_input('Lower Limit 15'))

                    #Scatter Plots Positive
                    st.write('This is cam 2 positive scatter plots')
                    x15 = cam2_df_positive['Battery No.']

                    #Display Plot
                    y15 = cam2_df_positive[option15]

                    plot15 = px.scatter(cam2_df_positive, x=x15, y=y15)
                    plot15.add_hline(y=ul15, line_width=3, line_color="red")
                    plot15.add_hline(y=ll15, line_width=3, line_color="red")
                    plot15.add_hrect(y0=ll15, y1=ul15, line_width=0, fillcolor="green", opacity=0.2)
                    st.plotly_chart(plot15, use_container_width=True)
                    
               with col16:
                    option16 = st.selectbox('Choose Y Axis', cam2_pos_col)
                    ul16 = int(st.number_input('Upper Limit 16'))
                    ll16 = int(st.number_input('Lower Limit 16'))

                    #Scatter Plots Positive
                    st.write('This is cam 2 positive scatter plots')
                    x16 = cam2_df_positive['Battery No.']

                    #Display Plot
                    y16 = cam2_df_positive[option16]

                    plot16 = px.scatter(cam2_df_positive, x=x16, y=y16)
                    plot16.add_hline(y=ul16, line_width=3, line_color="red")
                    plot16.add_hline(y=ll16, line_width=3, line_color="red")
                    plot16.add_hrect(y0=ll16, y1=ul16, line_width=0, fillcolor="green", opacity=0.2)
                    st.plotly_chart(plot16, use_container_width=True)
     
     #2PN Selected
     elif model_2PN_selected == True:
          if cam1_selected == True:
               #Import Cam 1 Data
               cam1_df = df

               #Cam 1 Data Sorting & Pre-processing
               cam1_df = cam1_df.sort_index(axis=1, ascending=True)
               cam1_df = cam1_df.sort_values(by=['Count'] , ascending=True)

               cam1_df_negative = cam1_df.loc[((cam1_df['Count'] % 10) < 6) & ((cam1_df['Count'] % 10) > 0)]
               cam1_df_negative = cam1_df_negative.drop(cam1_df_negative.iloc[:, 1:21],axis = 1)
               cam1_df_negative = cam1_df_negative.drop(cam1_df_negative.iloc[:,30:],axis = 1)

               cam1_df_positive = cam1_df.loc[((cam1_df['Count'] % 10) > 5) | ((cam1_df['Count'] % 10) == 0)]
               cam1_df_positive = cam1_df_positive.drop(cam1_df_positive.iloc[:, 21:],axis = 1)

               #Add Battery No
               cam1_df_negative['Battery No.'] = np.resize(samples_array, cam1_df_negative.shape[0])
               cam1_df_positive['Battery No.'] = np.resize(samples_array, cam1_df_positive.shape[0])

               st.write('This is cam 1 negative dataframe')
               st.dataframe(cam1_df_negative)

               st.write('This is cam 1 positive dataframe')
               st.dataframe(cam1_df_positive)

               #Scatter Plot Selection Negative
               cam1_neg_col = cam1_df_negative.columns.values.tolist()[1:-1]
               col1, col2, col3, col4 = st.columns(4)
               
               with col1:
                    option1 = st.selectbox('Choose Y Axis', cam1_neg_col)
                    ul1 = int(st.number_input('Upper Limit 1'))
                    ll1 = int(st.number_input('Lower Limit 1'))

                    #Scatter Plots Negative
                    st.write('This is cam 1 negative scatter plots')
                    x1 = cam1_df_negative['Battery No.']

                    #Display Plot
                    y1 = cam1_df_negative[option1]

                    plot1 = px.scatter(cam1_df_negative, x=x1, y=y1)
                    plot1.add_hline(y=ul1, line_width=3, line_color="red")
                    plot1.add_hline(y=ll1, line_width=3, line_color="red")
                    plot1.add_hrect(y0=ll1, y1=ul1, line_width=0, fillcolor="green", opacity=0.2)
                    st.plotly_chart(plot1, use_container_width=True)
                    
               with col2:
                    option2 = st.selectbox('Choose Y Axis', cam1_neg_col)
                    ul2 = int(st.number_input('Upper Limit 2'))
                    ll2 = int(st.number_input('Lower Limit 2'))

                    #Scatter Plots Negative
                    st.write('This is cam 1 negative scatter plots')
                    x2 = cam1_df_negative['Battery No.']

                    #Display Plot
                    y2 = cam1_df_negative[option2]

                    plot2 = px.scatter(cam1_df_negative, x=x2, y=y2)
                    plot2.add_hline(y=ul2, line_width=3, line_color="red")
                    plot2.add_hline(y=ll2, line_width=3, line_color="red")
                    plot2.add_hrect(y0=ll2, y1=ul2, line_width=0, fillcolor="green", opacity=0.2)
                    st.plotly_chart(plot2, use_container_width=True)
                    
               with col3:
                    option3 = st.selectbox('Choose Y Axis', cam1_neg_col)
                    ul3 = int(st.number_input('Upper Limit 3'))
                    ll3 = int(st.number_input('Lower Limit 3'))

                    #Scatter Plots Negative
                    st.write('This is cam 1 negative scatter plots')
                    x3 = cam1_df_negative['Battery No.']

                    #Display Plot
                    y3 = cam1_df_negative[option3]

                    plot3 = px.scatter(cam1_df_negative, x=x3, y=y3)
                    plot3.add_hline(y=ul3, line_width=3, line_color="red")
                    plot3.add_hline(y=ll3, line_width=3, line_color="red")
                    plot3.add_hrect(y0=ll3, y1=ul3, line_width=0, fillcolor="green", opacity=0.2)
                    st.plotly_chart(plot3, use_container_width=True)
                    
               with col4:
                    option4 = st.selectbox('Choose Y Axis', cam1_neg_col)
                    ul4 = int(st.number_input('Upper Limit 4'))
                    ll4 = int(st.number_input('Lower Limit 4'))

                    #Scatter Plots Negative
                    st.write('This is cam 1 negative scatter plots')
                    x4 = cam1_df_negative['Battery No.']

                    #Display Plot
                    y4 = cam1_df_negative[option4]

                    plot4 = px.scatter(cam1_df_negative, x=x4, y=y4)
                    plot4.add_hline(y=ul4, line_width=3, line_color="red")
                    plot4.add_hline(y=ll4, line_width=3, line_color="red")
                    plot4.add_hrect(y0=ll4, y1=ul4, line_width=0, fillcolor="green", opacity=0.2)
                    st.plotly_chart(plot4, use_container_width=True)
               

               #Scatter Plot Selection Positive
               col5, col6, col7, col8 = st.columns(4)
               cam1_pos_col = cam1_df_positive.columns.values.tolist()[1:-1]
               
               with col5:
                    option5 = st.selectbox('Choose Y Axis', cam1_pos_col)
                    ul5 = int(st.number_input('Upper Limit 5'))
                    ll5 = int(st.number_input('Lower Limit 5'))

                    #Scatter Plots Positive
                    st.write('This is cam 1 positive scatter plots')
                    x5 = cam1_df_positive['Battery No.']

                    #Display Plot
                    y5 = cam1_df_positive[option5]

                    plot5 = px.scatter(cam1_df_positive, x=x5, y=y5)
                    plot5.add_hline(y=ul5, line_width=3, line_color="red")
                    plot5.add_hline(y=ll5, line_width=3, line_color="red")
                    plot5.add_hrect(y0=ll5, y1=ul5, line_width=0, fillcolor="green", opacity=0.2)
                    st.plotly_chart(plot5, use_container_width=True)
                    
               with col6:
                    option6 = st.selectbox('Choose Y Axis', cam1_pos_col)
                    ul6 = int(st.number_input('Upper Limit 6'))
                    ll6 = int(st.number_input('Lower Limit 6'))

                    #Scatter Plots Positive
                    st.write('This is cam 1 positive scatter plots')
                    x6 = cam1_df_positive['Battery No.']

                    #Display Plot
                    y6 = cam1_df_positive[option6]

                    plot6 = px.scatter(cam1_df_positive, x=x6, y=y6)
                    plot6.add_hline(y=ul6, line_width=3, line_color="red")
                    plot6.add_hline(y=ll6, line_width=3, line_color="red")
                    plot6.add_hrect(y0=ll6, y1=ul6, line_width=0, fillcolor="green", opacity=0.2)
                    st.plotly_chart(plot6, use_container_width=True)
                    
               with col7:
                    option7 = st.selectbox('Choose Y Axis', cam1_pos_col)
                    ul7 = int(st.number_input('Upper Limit 7'))
                    ll7 = int(st.number_input('Lower Limit 7'))

                    #Scatter Plots Positive
                    st.write('This is cam 1 positive scatter plots')
                    x7 = cam1_df_positive['Battery No.']

                    #Display Plot
                    y7 = cam1_df_positive[option7]

                    plot7 = px.scatter(cam1_df_positive, x=x7, y=y7)
                    plot7.add_hline(y=ul7, line_width=3, line_color="red")
                    plot7.add_hline(y=ll7, line_width=3, line_color="red")
                    plot7.add_hrect(y0=ll7, y1=ul7, line_width=0, fillcolor="green", opacity=0.2)
                    st.plotly_chart(plot7, use_container_width=True)
                    
               with col8:
                    option8 = st.selectbox('Choose Y Axis', cam1_pos_col)
                    ul8 = int(st.number_input('Upper Limit 8'))
                    ll8 = int(st.number_input('Lower Limit 8'))

                    #Scatter Plots Positive
                    st.write('This is cam 1 positive scatter plots')
                    x8 = cam1_df_positive['Battery No.']

                    #Display Plot
                    y8 = cam1_df_positive[option8]

                    plot8 = px.scatter(cam1_df_positive, x=x8, y=y8)
                    plot8.add_hline(y=ul8, line_width=3, line_color="red")
                    plot8.add_hline(y=ll8, line_width=3, line_color="red")
                    plot8.add_hrect(y0=ll8, y1=ul8, line_width=0, fillcolor="green", opacity=0.2)
                    st.plotly_chart(plot8, use_container_width=True)

          elif cam2_selected == True:
               #Import Cam 2 Data
               cam2_df = df

               #Cam 2 Data Sorting & Pre-processing
               cam2_df = cam2_df.sort_index(axis=1, ascending=True)
               cam2_df = cam2_df.sort_values(by=['Count'] , ascending=True)

               cam2_df_negative = cam2_df.loc[((cam2_df['Count'] % 10) < 6) & ((cam2_df['Count'] % 10) > 0)]
               cam2_df_negative = cam2_df_negative.drop(cam2_df_negative.iloc[:, 1:69],axis = 1)

               cam2_df_positive = cam2_df.loc[((cam2_df['Count'] % 10) > 5) | ((cam2_df['Count'] % 10) == 0)]
               cam2_df_positive = cam2_df_positive.drop(cam2_df_positive.iloc[:, 1:51],axis = 1)
               cam2_df_positive = cam2_df_positive.drop(cam2_df_positive.iloc[:, 18:],axis = 1)

               #Add Battery No.
               cam2_df_negative['Battery No.'] = np.resize(samples_array, cam2_df_negative.shape[0])
               cam2_df_positive['Battery No.'] = np.resize(samples_array, cam2_df_positive.shape[0])

               st.write('This is cam 2 negative dataframe')
               st.dataframe(cam2_df_negative)

               st.write('This is cam 2 positive dataframe')
               st.dataframe(cam2_df_positive)

               #Scatter Plot Selection Negative
               cam2_neg_col = cam2_df_negative.columns.values.tolist()[1:-1]
               col9, col10, col11, col12 = st.columns(4)
               
               with col9:
                    option9 = st.selectbox('Choose Y Axis', cam2_neg_col)
                    ul9 = int(st.number_input('Upper Limit 9'))
                    ll9 = int(st.number_input('Lower Limit 9'))

                    #Scatter Plots Negative
                    st.write('This is cam 2 negative scatter plots')
                    x9 = cam2_df_negative['Battery No.']

                    #Display Plot
                    y9 = cam2_df_negative[option9]

                    plot9 = px.scatter(cam2_df_negative, x=x9, y=y9)
                    plot9.add_hline(y=ul9, line_width=3, line_color="red")
                    plot9.add_hline(y=ll9, line_width=3, line_color="red")
                    plot9.add_hrect(y0=ll9, y1=ul9, line_width=0, fillcolor="green", opacity=0.2)
                    st.plotly_chart(plot9, use_container_width=True)
                    
               with col10:
                    option10 = st.selectbox('Choose Y Axis', cam2_neg_col)
                    ul10 = int(st.number_input('Upper Limit 10'))
                    ll10 = int(st.number_input('Lower Limit 10'))

                    #Scatter Plots Negative
                    st.write('This is cam 2 negative scatter plots')
                    x10 = cam2_df_negative['Battery No.']

                    #Display Plot
                    y10 = cam2_df_negative[option10]

                    plot10 = px.scatter(cam2_df_negative, x=x10, y=y10)
                    plot10.add_hline(y=ul10, line_width=3, line_color="red")
                    plot10.add_hline(y=ll10, line_width=3, line_color="red")
                    plot10.add_hrect(y0=ll10, y1=ul10, line_width=0, fillcolor="green", opacity=0.2)
                    st.plotly_chart(plot10, use_container_width=True)
                    
               with col11:
                    option11 = st.selectbox('Choose Y Axis', cam2_neg_col)
                    ul11 = int(st.number_input('Upper Limit 11'))
                    ll11 = int(st.number_input('Lower Limit 11'))

                    #Scatter Plots Negative
                    st.write('This is cam 2 negative scatter plots')
                    x11 = cam2_df_negative['Battery No.']

                    #Display Plot
                    y11 = cam2_df_negative[option11]

                    plot11 = px.scatter(cam2_df_negative, x=x11, y=y11)
                    plot11.add_hline(y=ul11, line_width=3, line_color="red")
                    plot11.add_hline(y=ll11, line_width=3, line_color="red")
                    plot11.add_hrect(y0=ll11, y1=ul11, line_width=0, fillcolor="green", opacity=0.2)
                    st.plotly_chart(plot11, use_container_width=True)
                    
               with col12:
                    option12 = st.selectbox('Choose Y Axis', cam2_neg_col)
                    ul12 = int(st.number_input('Upper Limit 12'))
                    ll12 = int(st.number_input('Lower Limit 12'))

                    #Scatter Plots Negative
                    st.write('This is cam 2 negative scatter plots')
                    x12 = cam2_df_negative['Battery No.']

                    #Display Plot
                    y12 = cam2_df_negative[option12]

                    plot12 = px.scatter(cam2_df_negative, x=x12, y=y12)
                    plot12.add_hline(y=ul12, line_width=3, line_color="red")
                    plot12.add_hline(y=ll12, line_width=3, line_color="red")
                    plot12.add_hrect(y0=ll12, y1=ul12, line_width=0, fillcolor="green", opacity=0.2)
                    st.plotly_chart(plot12, use_container_width=True)

               #Scatter Plot Selection Positive
               cam2_pos_col = cam2_df_positive.columns.values.tolist()[1:-1]
               col13, col14, col15, col16 = st.columns(4)
               
               with col13:
                    option13 = st.selectbox('Choose Y Axis', cam2_pos_col)
                    ul13 = int(st.number_input('Upper Limit 13'))
                    ll13 = int(st.number_input('Lower Limit 13'))

                    #Scatter Plots Positive
                    st.write('This is cam 2 positive scatter plots')
                    x13 = cam2_df_positive['Battery No.']

                    #Display Plot
                    y13 = cam2_df_positive[option13]

                    plot13 = px.scatter(cam2_df_positive, x=x13, y=y13)
                    plot13.add_hline(y=ul13, line_width=3, line_color="red")
                    plot13.add_hline(y=ll13, line_width=3, line_color="red")
                    plot13.add_hrect(y0=ll13, y1=ul13, line_width=0, fillcolor="green", opacity=0.2)
                    st.plotly_chart(plot13, use_container_width=True)
                    
               with col14:
                    option14 = st.selectbox('Choose Y Axis', cam2_pos_col)
                    ul14 = int(st.number_input('Upper Limit 14'))
                    ll14 = int(st.number_input('Lower Limit 14'))

                    #Scatter Plots Positive
                    st.write('This is cam 2 positive scatter plots')
                    x14 = cam2_df_positive['Battery No.']

                    #Display Plot
                    y14 = cam2_df_positive[option14]

                    plot14 = px.scatter(cam2_df_positive, x=x14, y=y14)
                    plot14.add_hline(y=ul14, line_width=3, line_color="red")
                    plot14.add_hline(y=ll14, line_width=3, line_color="red")
                    plot14.add_hrect(y0=ll14, y1=ul14, line_width=0, fillcolor="green", opacity=0.2)
                    st.plotly_chart(plot14, use_container_width=True)
                    
               with col15:
                    option15 = st.selectbox('Choose Y Axis', cam2_pos_col)
                    ul15 = int(st.number_input('Upper Limit 15'))
                    ll15 = int(st.number_input('Lower Limit 15'))

                    #Scatter Plots Positive
                    st.write('This is cam 2 positive scatter plots')
                    x15 = cam2_df_positive['Battery No.']

                    #Display Plot
                    y15 = cam2_df_positive[option15]

                    plot15 = px.scatter(cam2_df_positive, x=x15, y=y15)
                    plot15.add_hline(y=ul15, line_width=3, line_color="red")
                    plot15.add_hline(y=ll15, line_width=3, line_color="red")
                    plot15.add_hrect(y0=ll15, y1=ul15, line_width=0, fillcolor="green", opacity=0.2)
                    st.plotly_chart(plot15, use_container_width=True)
                    
               with col16:
                    option16 = st.selectbox('Choose Y Axis', cam2_pos_col)
                    ul16 = int(st.number_input('Upper Limit 16'))
                    ll16 = int(st.number_input('Lower Limit 16'))

                    #Scatter Plots Positive
                    st.write('This is cam 2 positive scatter plots')
                    x16 = cam2_df_positive['Battery No.']

                    #Display Plot
                    y16 = cam2_df_positive[option16]

                    plot16 = px.scatter(cam2_df_positive, x=x16, y=y16)
                    plot16.add_hline(y=ul16, line_width=3, line_color="red")
                    plot16.add_hline(y=ll16, line_width=3, line_color="red")
                    plot16.add_hrect(y0=ll16, y1=ul16, line_width=0, fillcolor="green", opacity=0.2)
                    st.plotly_chart(plot16, use_container_width=True)

















