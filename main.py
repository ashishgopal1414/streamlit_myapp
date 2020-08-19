# Core Packages
import io, os, shutil
from os import path
import missingno as msno

import streamlit as st

# EDA Packages
import pandas as pd
import numpy as numpy

# Data Viz Packages
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
import seaborn as sns
import missingno as msno
from pandas_profiling import ProfileReport

# Machine Learning Packages
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
import sklearn

## Disable Warning
st.set_option('deprecation.showfileUploaderEncoding', False)
#%%

data_flag = 0

#%%
current_path = os.getcwd()

## Create sub directories if not created: "Raw Data" , "Batch Wise Data" , "Aggregated Data"
folder_names = [name for name in ["Raw Data" , "Modified Data"]]

for folder_name in folder_names:
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
#%%

datafile_path = os.path.join(current_path, "Raw Data", "data.csv")
modifiedfile_path = os.path.join(current_path, "Modified Data", "data.csv")

data_df = pd.DataFrame()

################################################################################
# from typing import Dict
# @st.cache(allow_output_mutation=True)
# def get_static_store() -> Dict:
#     """This dictionary is initialized once and can be used to store the files uploaded"""
#     return {}

################################################################################

##@st.cache(suppress_st_warning=True)
def load_data():
    data_df = pd.DataFrame()
    if path.exists(datafile_path):
        data_df = pd.read_csv(datafile_path)
        if st.checkbox("Click to view data"):
            st.write(data_df)
    return data_df

################################################################################

def load_modified_data():
    data_df = pd.DataFrame()
    if (not os.path.exists(datafile_path)) & (os.path.exists(modifiedfile_path)):
        os.remove(modifiedfile_path)
    if path.exists(modifiedfile_path):
        data_df = pd.read_csv(modifiedfile_path)
        if st.checkbox("Click to view Modified data"):
            st.write(data_df)
    return data_df
################################################################################
from streamlit_pandas_profiling import st_profile_report

#@st.cache(suppress_st_warning=True)
def profile_report(df):
    report = ProfileReport(df, minimal=True)
    # components.v1.html(report.to_html())
    st_profile_report(report)
################################################################################

def summary_data(df):
    st.write('---------------------------------------------------')
    if not df.empty:
        # Details of Data
        st.write('#######################################')
        if st.checkbox("Top 20 rows"):
            st.dataframe(df.head(20))
        if st.checkbox("Bottom 20 rows"):
            st.dataframe(df.tail(20))
        # Show Shape
        if st.checkbox("Show Shape"):
            st.write(df.shape)
        if st.checkbox("Show Columns"):
            all_columns = df.columns = df.columns.to_list()
            st.write(all_columns)
        if st.checkbox("Show Summary"):
            st.write(df.describe(include='all'))
        if st.checkbox("Show Column Datatypes"):
            st.write(df.dtypes)
        if st.checkbox("Data Profiling (Takes few minutes to execute)"):
            profile_report(df)

        st.write('#######################################')
    else:
        st.markdown('**No Data Available to show!**.')
    st.write('---------------------------------------------------')

################################################################################

def preprocess_data(data_df):
    st.write('---------------------------------------------------')
    if not data_df.empty:
        all_columns = data_df.columns.to_list()
        if st.checkbox("Data Preprocess (Keep checked in to add steps)"):
            ## Receive a function to be called for Preprocessing
            df = data_df.copy()
            txt = st.text_area(
                "Provide lines of code in the given format to preprocess the data, otherwise leave it as commented",
                "## Consider the dataframe to be stored in 'df' variable\n" + \
                "## for e.g.\n" + \
                "## df['col_1'] = df['col_1'].astype('str')")
            if st.button("Finally, Click here to update the file"):
                exec(txt)
                if os.path.exists(modifiedfile_path):
                    os.remove(modifiedfile_path)
                df.to_csv(modifiedfile_path, index=False)
                st.success("New file created successfully under: {}".format(modifiedfile_path))
            if st.checkbox("Click to view Modified file"):
                if os.path.exists(modifiedfile_path):
                    st.write(pd.read_csv(modifiedfile_path))
                else:
                    st.markdown('**No Data Available to show!**.')

    else:
        st.markdown('**No Data Available to show!**.')
    st.write('---------------------------------------------------')

################################################################################

def visualization_data(df):
    df = df.copy()
    txt = "#"
    st.write('---------------------------------------------------')
    if not df.empty:
        all_columns = df.columns.to_list()
        if st.checkbox("NULL counts in Data"):
            selected_columns = st.multiselect("Select Columns ", all_columns)
            if len(selected_columns) > 0:
                new_df = df[selected_columns].copy()
                msno.matrix(new_df)
                st.pyplot()
        if st.checkbox("Visualize a particular column"):
            st.subheader("Data Preprocessing step")
            ## Receive a function to be called for Preprocessing
            txt = st.text_area(
                "Provide lines of code in the given format to preprocess the data",
                "## Consider the dataframe to be stored in 'df' variable\n" + \
                "## for e.g.\n" + \
                "## df['col_1'] = df['col_1'].astype('str')")
            plot_col_x = st.selectbox("Select Columns for X axis", all_columns)
            plot_col_y = st.multiselect("Select Columns for Y axis", all_columns)
            if (plot_col_x is not None) | (plot_col_y is not None):

                exec(txt)
                # Raw data plot of Mositure variable
                st.write("{} with respect to {}".format(plot_col_x, plot_col_y))
                plt.figure(figsize=(18, 4))
                plt.xlabel(plot_col_x)
                #plt.xticks(df[plot_col_x])
                columns_selected = [plot_col_x] + plot_col_y
                st.write(df[columns_selected])
                for i in range(len(plot_col_y)):
                    plt.plot(df[plot_col_y[i]], label=plot_col_y)
                plt.legend()
                #plt.ylabel(plot_col_y)
                st.pyplot()
            else:
                st.write("Please Select Columns")
    else:
        st.markdown('**No Data Available to show!**.')
        st.write("Did you run the Data Preprocess Step? if not, first run and then try again.")
    st.write('---------------------------------------------------')

################################################################################
# import SessionState
import time

def main():
    """ Semi Supervised Machine Learning App with Streamlit """
    # static_store = get_static_store()
    # session = SessionState.get(run_id=0)

    st.title("Complete Data Science Application")
    #st.text("By Ashish Gopal")

    activities_outer = ["Data Ingestion", "Others", "About"]
    choice_1_outer = st.sidebar.radio("Choose your Step:", activities_outer)

    data = pd.DataFrame()

    if choice_1_outer == "Data Ingestion":
        file_types = ["csv","txt"]

        activities_1 = ["1. Data Import", "2. Data Summary", "3. Data Preprocess", "4. Data Visualization"]
        choice_1 = st.sidebar.selectbox("Select Activities", activities_1)

        if choice_1 == "1. Data Import":
            data = None
            show_file = st.empty()
            if st.checkbox("Click to Upload data"):
                data = st.file_uploader("Upload Dataset : ",type=file_types)
            if not data:
                show_file.info("Please upload a file of type: " + ", ".join(file_types))
                if os.path.exists(datafile_path):
                    os.remove(datafile_path)
                return
            if data:
                if st.button("Click to delete data"):
                    if os.path.exists(datafile_path):
                        os.remove(datafile_path)
                        st.write('Raw File deleted successfully!')
                    elif os.path.exists(modifiedfile_path):
                        os.remove(modifiedfile_path)
                        st.write('Modified File deleted successfully!')
                    else:
                        st.write('No Files available for deletion!')
                    # static_store.clear()
                    data = None
                    # session.run_id += 1
                    return

                if data is not None:
                    df = pd.read_csv(data)
                    df.to_csv(datafile_path, index=False)
                    st.write('File loaded successfully!')
                if st.checkbox("Click to view data"):
                    if data is not None:
                        st.write(df)
                    else:
                        st.write('No Data available!')

        if choice_1 == "2. Data Summary":
            summary_data(load_data())

        if choice_1 == "3. Data Preprocess":
            preprocess_data(load_data())

        if choice_1 == "4. Data Visualization":
            visualization_data(load_modified_data())
            if st.button('Submit'):
                for i in range(5):
                    st.balloons()
                    time.sleep(1)

    if choice_1_outer == "Others":
        st.write("Coming Soon... ")
        st.write('---------------------------------------------------')

    if choice_1_outer == "About":
        st.sidebar.header("About App")
        st.sidebar.info("Complete Data Science Cycle Application ")
        st.title("")
        st.title("")
        st.sidebar.header("About Developer")
        st.sidebar.info("https://www.linkedin.com/in/ashish-gopal-73824572/")
        st.subheader("About Me")
        st.text("Name: Ashish Gopal")
        st.text("Job Profile: Data Scientist")
        IMAGE_URL = "https://avatars0.githubusercontent.com/u/36658472?s=460&v=4"
        st.image(IMAGE_URL, use_column_width=True)
        st.markdown("LinkedIn: https://www.linkedin.com/in/ashish-gopal-73824572/")
        st.markdown("GitHub: https://github.com/ashishgopal1414")
        st.write('---------------------------------------------------')
if __name__ == '__main__':
	main()