# Import packages
import math
import datetime
import os.path

import pandas as pd


def load_df(dir_path="./data/covid/"):
    current_date = datetime.date.today().strftime("%Y%m%d")
    covid_mod_filename = "covid19_" + current_date + "_mod.csv"
    covid_mod_filepath = dir_path + covid_mod_filename
    if os.path.isfile(covid_mod_filepath):
        print("Current data exists, load processed datafile")
        covid_mod_data = pd.read_csv(covid_mod_filepath, parse_dates=["date"])

    else:
        print(
            "Current procssed datafile does not not exist, download raw data and process"
        )
        covid_raw_filename = "covid19_" + current_date + "_raw.csv"
        covid_raw_filepath = dir_path + covid_raw_filename
        if os.path.isfile(covid_raw_filepath):
            covid_raw_data = pd.read_csv(covid_raw_filepath, parse_dates=["date"])
        else:
            url = "https://health-infobase.canada.ca/src/data/covidLive/covid19-download.csv"
            covid_raw_data = pd.read_csv(url, parse_dates=["date"])
            covid_raw_data.to_csv(covid_raw_filepath, index=False, header=True)

        # process raw data
        covid_mod_data = process_raw_df(covid_raw_data)
        # export to processed data to csv
        covid_mod_data.to_csv(covid_mod_filepath, index=False, header=True)

    return covid_mod_data


def process_raw_df(df):
    covid_data = df.copy()
    # only look at BC and reset the index
    covid_data = covid_data[covid_data["prname"] == "British Columbia"].reset_index(
        drop=True
    )
    # only keep columns of interest
    covid_data = covid_data.loc[
        :,
        (
            "date",
            "numtotal",
            "numdeaths",
            "numrecover",
        ),
    ]

    # create a new column which takes the difference between cumulative case counts
    covid_data["new_count"] = covid_data["numtotal"] - covid_data["numtotal"].shift()
    # set the first day in new_count to numtotal val
    covid_data.loc[0, "new_count"] = covid_data.loc[0, "numtotal"]
    # also create new daily count columns for deaths and recover
    covid_data["numdeaths"].fillna(0, inplace=True)
    covid_data["new_deaths"] = covid_data["numdeaths"] - covid_data["numdeaths"].shift()
    covid_data.loc[0, "new_deaths"] = covid_data.loc[0, "numdeaths"]
    covid_data["numrecover"].fillna(0, inplace=True)
    covid_data["new_recover"] = (
        covid_data["numrecover"] - covid_data["numrecover"].shift()
    )
    covid_data.loc[0, "new_recover"] = covid_data.loc[0, "numrecover"]

    # change the date column to the index and the drop the date column
    covid_data.index = pd.DatetimeIndex(covid_data.date)
    covid_data.drop(columns="date", inplace=True)
    # create the complete date range
    idx = pd.date_range(covid_data.index.min(), covid_data.index.max())
    # update the index with the full date range and fill the new rows with zeros
    covid_data = covid_data.reindex(idx, fill_value=0)
    # move the date index to a column and create a new integer index
    covid_data.reset_index(inplace=True)
    covid_data.rename(columns={"index": "date"}, inplace=True)
    # Calculate days elapsed
    covid_data["days_elapse"] = (covid_data["date"] - covid_data["date"].min()).dt.days
    # Remove any negative values and replace with zero
    covid_data["new_recover"] = covid_data["new_recover"].apply(
        lambda x: x if x > 0 else 0
    )

    # remove any rows with zero counts at the end of the dataframe as they represent unreported values
    # use the counts column for this as it is assumed any days without counts would also not have counts for deaths and recovery
    # but possibly not the other way around
    max_index = len(covid_data) - 1
    print("inital dataframe length", max_index)

    # work backwards through the data frame removing rows until a non-zero count is encountered
    for i in range(max_index):
        rev_i = max_index - i
        if covid_data.loc[rev_i, "new_count"] == 0:
            covid_data.drop([rev_i], inplace=True)
            print("dropped row ", rev_i)
        # break the loop once a non-zero value is found
        else:
            break

    max_index = len(covid_data) - 1
    print("updated dataframe length", max_index)

    # create new columns for the modifications
    covid_data["new_count_update"] = covid_data["new_count"]
    covid_data["new_deaths_update"] = covid_data["new_deaths"]
    covid_data["new_recover_update"] = covid_data["new_recover"]

    # use the corrector function to realocate counts for unreported days as required
    for ind, row in covid_data.iterrows():
        # start above index 30 for counts and deaths as discussed above
        if ind > 30 and row["new_count_update"] == 0:
            corrector(covid_data, "new_count_update", ind)
        if ind > 30 and row["new_deaths_update"] == 0:
            corrector(covid_data, "new_deaths_update", ind)
        # start above index 55 for recoveries as discussed above
        if ind > 55 and row["new_recover_update"] == 0:
            corrector(covid_data, "new_recover_update", ind)

    # create new columns representing the cumulative values after the corrections above are applied
    covid_data["new_numtotal"] = covid_data["new_count_update"].cumsum()
    covid_data["new_numdeaths"] = covid_data["new_deaths_update"].cumsum()
    covid_data["new_numrecover"] = covid_data["new_recover_update"].cumsum()

    # Return a df with the columns of interest
    return covid_data[
        [
            "date",
            "days_elapse",
            "new_count_update",
            "new_deaths_update",
            "new_recover_update",
            "new_numtotal",
            "new_numdeaths",
            "new_numrecover",
        ]
    ]


# Function to smooth out values over the unreported dates
# NOTE THAT THIS FUNCTION PERFORMS INPLACE OPERATIONS!
def corrector(df, col_name, ind):
    # check if the column of interest has a zero value at the index
    if df.loc[ind, col_name] == 0:
        temp_list = [ind]
        counter = 1

        # continue looping forward and adding to the list for any zero values
        while df.loc[(ind + counter), col_name] == 0:
            temp_list.append(ind + counter)
            counter += 1

        # also append the first non-zero value as this will represent the cumulative count from the zero days
        temp_list.append(ind + counter)

        # the following calculations determine the average for the days
        temp_length = len(temp_list)
        old_value_count = df.loc[(temp_list[-1]), col_name]
        # only use complete integers (rounding down to ensure don't increase the count)
        new_values_count = math.floor(old_value_count / temp_length)
        # calculate the remainder from rounding downto ensure complete count is preserved
        remainder_count = old_value_count - (temp_length * new_values_count)

        # replacing the corresponding values stored in the temporary index list
        for ind_replace in temp_list:
            # if the last value (which is the non-zero count in the index list), add the remainder
            if ind_replace == temp_list[-1]:
                df.loc[ind_replace, col_name] = new_values_count + remainder_count
            else:
                df.loc[ind_replace, col_name] = new_values_count
