# Import packages
import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

url = "https://health-infobase.canada.ca/src/data/covidLive/covid19.csv"
covid_1 = pd.read_csv(url)
covid_1["date"] = pd.to_datetime(covid_1["date"], format="%d-%m-%Y")

# only look at BC and reset the index
# note there is already a column called index that retains the original dataframe index
covid_2 = (
    covid_1[covid_1["prname"] == "British Columbia"]
    .drop(columns=["prnameFR"])
    .copy()
    .reset_index()
)
covid_2.rename(columns={"index": "original_index"}, inplace=True)

# add a new column for days elapsed since start date
covid_2["days_elapse"] = (covid_2["date"] - covid_2["date"].min()).dt.days

# create a new column which takes the difference between cumulative case counts
covid_2["new_count"] = covid_2["numtotal"] - covid_2["numtotal"].shift()
covid_2.loc[0, "new_count"] = covid_2.loc[0, "numtotal"]

# create a check column which assess if "new_count" is the same as numtoday
covid_2["new_count_check"] = covid_2["new_count"] == covid_2["numtoday"]

# based on the above, decide to use the new_count column as the daily values
# this will ensure we're preserving the cumulative total count

# also create new daily count columns for deaths and recover
covid_2["numdeaths"].fillna(0, inplace=True)
covid_2["new_deaths"] = covid_2["numdeaths"] - covid_2["numdeaths"].shift()
covid_2.loc[0, "new_deaths"] = covid_2.loc[0, "numdeaths"]

covid_2["numrecover"].fillna(0, inplace=True)
covid_2["new_recover"] = covid_2["numrecover"] - covid_2["numrecover"].shift()
covid_2.loc[0, "new_recover"] = covid_2.loc[0, "numrecover"]

# change the date column to the index and the drop the date column
covid_2.index = pd.DatetimeIndex(covid_2.date)
covid_2.drop(columns="date", inplace=True)
# create the complete date range
idx = pd.date_range(covid_2.index.min(), covid_2.index.max())
# update the index with the full date range and fill the new rows with zeros
covid_2 = covid_2.reindex(idx, fill_value=0)
# move the date index to a column and create a new integer index
covid_2.reset_index(inplace=True)
covid_2.rename(columns={"index": "date"}, inplace=True)

# Need to recalculate days elapsed to populate new date rows
covid_2["days_elapse"] = (covid_2["date"] - covid_2["date"].min()).dt.days

# create a copy of the dataframe for modifications
covid_3 = covid_2.copy()

# create new columns for the modifications
covid_3["new_count_update"] = covid_3["new_count"]
covid_3["new_deaths_update"] = covid_3["new_deaths"]
covid_3["new_recover_update"] = covid_3["new_recover"]

# remove any rows with zero counts at the end of the dataframe as they represent unreported values
# use the counts column for this as it is assumed any days without counts would also not have counts for deaths and recovery
# but possibly not the other way around
max_index = len(covid_3) - 1
print("inital dataframe length", max_index)

# work backwards through the data frame removing rows until a non-zero count in encountered
for i in range(max_index):
    rev_i = max_index - i
    if covid_3.loc[rev_i, "new_count"] == 0:
        covid_3.drop([rev_i], inplace=True)
        print("dropped row ", rev_i)
    # break the loop once a non-zero value is found
    else:
        break

max_index = len(covid_3) - 1
print("updated dataframe length", max_index)
