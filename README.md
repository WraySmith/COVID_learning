# COVID Learning Repo

This repo is being used as a self-learning tool using publicly available [Canada COVID-19 Dataset](https://open.canada.ca/data/en/dataset/261c32ab-4cfd-4f81-9dea-7b64065690dc).  The purpose of the work is for learning and currently includes the following in progress:

- **Web scraping**:
    - Largely complete, sentiment analysis didn't indicate any trend (note that this only includes data up to early July)
    - *compile_urls.ipynb*: compiles CBC article URLs that reference COVID-19 in BC
    - *read_articles.ipynb*: parses the articles in the the URLs from compile_urls.ipynb and runs a sentiment analysis on each article
    - *plot_articles.ipynb*: plots the sentiment analysis against time to assess trends
- **ARIMA**:
    - Largely complete, as expected the ARIMA analysis isn't that useful for prediction of BC cases (largely appears to be a random walk process from a statistical sense) - scripts need a bit of clean-up still
    - *covid_explore.ipynb*: explore the COVID-19 data available for BC and preprocess the data for use with an ARIMA analysis
    - *covid_stats.ipynb*: ARIMA analysis of data preprocessed in covid_explore.ipynb
- **Streamlit**
    - Create a Streamlit app of the ARIMA analysis - still largely in progress
    - *covid_app.py*: main body of the streamlit app
    - *covid_app_func.py*: functions for app
