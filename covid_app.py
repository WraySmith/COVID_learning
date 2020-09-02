import streamlit as st
import covid_app_func as func
import matplotlib.pyplot as plt

# Load Data
def load_data():
    data_path = './data/covid/covid19_20200725_mod.csv'
    column_interest = 'new_count_update'
    n_mean = 3
    df = func.process_data(data_path, column_interest)
    df_out, plot_out = func.mean_data(df, n_mean)
    df_out, lam_out = func.add_series(df_out, columns_analyze = 'daily_case')
    return df_out, lam_out, plot_out

def main():
    st.title('BC COVID Cases Explorer')
    # Add a selectbox to the sidebar:
    select_menu = st.sidebar.selectbox(
        'Choose Page',
        ('Working', 'Explore')
    )
    covid_data, lam_print, mean_plot = load_data()

    if select_menu == "Explore":
        st.pyplot(mean_plot)
        st.text('Lambda of Box-Cox Transform: ' + str(lam_print))
        st.write(covid_data.head())
        col_explore = ['daily_case',  'daily_case_diff', 'daily_case_bc', \
            'daily_case_bc_diff', 'daily_case_2wknormal', 'daily_case_2wknormal_diff']
        explore_data, explore_plot = func.explore_series(covid_data, col_explore)
        st.pyplot(explore_plot)
        st.write(explore_data)


if __name__ == "__main__":
    main()








