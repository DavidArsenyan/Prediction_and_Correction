import pandas as pd
import re
from random import randint
# Define the regex pattern for a valid tavg value

def corr_data(data):

    pattern = re.compile(r"\d\d.\d")

    # Data is CSV file

    # Initialize lists to store valid and invalid rows
    valid_rows = []
    invalid_rows = []

    # Iterate through the rows and validate the tavg values
    for index, row in data.iterrows():
        tavg = str(row['tavg'])

        if pattern.match(tavg):
            valid_rows.append(row)
        else:
            invalid_rows.append(row)


    # Convert valid and invalid rows back to DataFrames
    valid_df = pd.DataFrame(valid_rows)
    invalid_df = pd.DataFrame(invalid_rows)

    # Output the results
    return invalid_df

def pattern(df, db_predicted):
    tavg = df["tavg"].values
    null_tavg = []
    for i in range(len(df["tavg"].values)):
        if len(tavg[i]) == 3:
            null_tavg += [i]
        else:
            cur_sym = ""
            rand_num = randint(0, 9)
            for j in range(len(tavg[i])):
                cur_symbol = tavg[i][j]
                if re.compile(r"\d|\.").match(cur_symbol):
                    cur_sym += str(tavg[i][j])
                else:
                    if j == 0:
                        rand_num = randint(1, 2)
                    cur_sym += str(rand_num)

            tavg[i] = cur_sym

    for i in null_tavg:
        tavg[i] = db_predicted["tavg"].values[i]

    # Ensure all tavg values are numeric (convert to float)
    return [float(x) for x in tavg]
