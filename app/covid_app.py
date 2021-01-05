import streamlit as st
import covid_app_process as process
import matplotlib.pyplot as plt
import warnings


def main():
    st.title("BC COVID Cases Explorer")  # app title
    df_initial = process.load_df("")  # read data

    select_menu = st.sidebar.selectbox(  # select overall app menu
        "Choose Page", ("Explore Data", "Other")
    )


if __name__ == "__main__":
    main()
