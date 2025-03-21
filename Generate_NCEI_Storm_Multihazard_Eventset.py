#######################
"""
Created March 21st, 2025

@author: Joshua Green - University of Southampton

Please cite this dataset if used in any publications.

Green, J. (2025) NCEI Storm Multihazard Eventset.
"""
#######################

import datetime
import pandas as pd
import numpy as np
from datetime import datetime
import os
from tqdm import tqdm

# pd.set_option('display.max_colwidth', None)
# pd.set_option('display.max_columns', None)

Cleaned_NCEI_Storm_Database_Parquet_Path = r"PATH GOES HERE"
Output_Eventset_Path = r"PATH GOES HERE"

# Define temporal year range
# CHANGE THESE VALUES AS DESIRED FOR TEMPORAL COVERAGE
start_year = 1996
end_year = 2024

# Define time lag in days
# CHANGE THESE VALUES AS DESIRED FOR APPROPRIATE TEMPORAL OVERLAP
time_lag = pd.Timedelta(days=90)
time_lag_int = time_lag.days


raw_df = pd.read_parquet(Cleaned_NCEI_Storm_Database_Parquet_Path)
# df['BEGIN_DATETIME'] = pd.to_datetime(df['BEGIN_DATETIME'])
# df['END_DATETIME'] = pd.to_datetime(df['END_DATETIME'])
dfsingle = raw_df

dfsingle["CZ_FIPS"] = dfsingle["CZ_FIPS"].astype(str).str.zfill(3)
dfsingle["STATE_FIPS"] = dfsingle["STATE_FIPS"].astype(str).str.zfill(2)
dfsingle["GEOID"] = dfsingle["STATE_FIPS"].astype(str).str.zfill(2) + dfsingle[
    "CZ_FIPS"
].astype(str).str.zfill(3)

dfsingle = dfsingle[~dfsingle["EPISODE_ID"].isnull()]
dfsingle = dfsingle[~dfsingle["EVENT_ID"].isnull()]
dfsingle = dfsingle[~dfsingle["STATE"].isnull()]
dfsingle = dfsingle[~dfsingle["STATE_FIPS"].isnull()]
dfsingle = dfsingle[~dfsingle["EVENT_TYPE"].isnull()]
dfsingle = dfsingle[~dfsingle["CZ_FIPS"].isnull()]
dfsingle = dfsingle[~dfsingle["BEGIN_DATETIME"].isnull()]
dfsingle = dfsingle[~dfsingle["END_DATETIME"].isnull()]

# remove these columns if desired
# dfsingle = dfsingle.drop(columns=['EPISODE_NARRATIVE', 'EVENT_NARRATIVE'])

dfsingle["DAMAGE_CROPS"] = dfsingle["DAMAGE_CROPS"].fillna(0).astype(int)
dfsingle["DAMAGE_PROPERTY"] = dfsingle["DAMAGE_PROPERTY"].fillna(0).astype(int)
dfsingle = dfsingle.drop_duplicates()

# dfsingle['start_year'] = dfsingle['BEGIN_DATETIME'].dt.year
# dfsingle['end_year'] = dfsingle['END_DATETIME'].dt.year

dfsingle = dfsingle.reindex(
    columns=[
        "EPISODE_ID",
        "EVENT_ID",
        "GEOID",
        "STATE",
        "STATE_FIPS",
        "EVENT_TYPE",
        "HAZARD",
        "CZ_TYPE",
        "CZ_FIPS",
        "CZ_NAME",
        "BEGIN_DATETIME",
        "END_DATETIME",
        "start_year",
        "end_year",
        "WFO",
        "CZ_TIMEZONE",
        "INJURIES_DIRECT",
        "INJURIES_INDIRECT",
        "DEATHS_DIRECT",
        "DEATHS_INDIRECT",
        "DAMAGE_PROPERTY",
        "DAMAGE_CROPS",
        "SOURCE",
        "MAGNITUDE",
        "MAGNITUDE_TYPE",
        "FLOOD_CAUSE",
        "CATEGORY",
        "TOR_F_SCALE",
        "TOR_LENGTH",
        "TOR_WIDTH",
        "TOR_OTHER_WFO",
        "TOR_OTHER_CZ_STATE",
        "TOR_OTHER_CZ_FIPS",
        "TOR_OTHER_CZ_NAME",
        "BEGIN_RANGE",
        "BEGIN_AZIMUTH",
        "BEGIN_LOCATION",
        "END_RANGE",
        "END_AZIMUTH",
        "END_LOCATION",
        "BEGIN_LAT",
        "BEGIN_LON",
        "END_LAT",
        "END_LON",
        "DATA_SOURCE",
        "EPISODE_NARRATIVE",
        "EVENT_NARRATIVE",
    ]
)

# temporal filter
dfsingle = dfsingle[
    (dfsingle["BEGIN_DATETIME"].dt.year >= start_year)
    & (dfsingle["END_DATETIME"].dt.year <= end_year)
]

# remove unwanted state classes
# CHANGE THE EXCLUSED STATES AS DESIRED
# NOTE THAT THIS CLASSIFICATION INCLUDES US TERRITORIES AND WATER BODIES
Exclusion_State_List = [
    "ATLANTIC SOUTH",
    "LAKE ST CLAIR",
    "ALASKA",
    "GUAM",
    "LAKE MICHIGAN",
    "LAKE ONTARIO",
    "PUERTO RICO",
    "GULF OF MEXICO",
    "ATLANTIC NORTH",
    "VIRGIN ISLANDS",
    "LAKE HURON",
    "AMERICAN SAMOA",
    "ST LAWRENCE R",
    "LAKE ERIE",
    "LAKE SUPERIOR",
    "E PACIFIC",
]
dfsingle = dfsingle[~dfsingle["STATE"].isin(Exclusion_State_List)]

# impact filter thresholds
# CHANGE THESE VALUES AS DESIRED FOR APPROPRIATE IMPACT FILTERING
inj = 1
dth = 1
c = 50  # in thoughsands
p = 50  # in thoughsands

# filter by event impact
dfsingle = dfsingle[
    (
        (dfsingle["INJURIES_DIRECT"] >= inj)
        | (dfsingle["INJURIES_INDIRECT"] >= inj)
        | (dfsingle["DEATHS_DIRECT"] >= dth)
        | (dfsingle["DEATHS_INDIRECT"] >= dth)
        | (dfsingle["DAMAGE_CROPS"] >= c * 1000)
        | (dfsingle["DAMAGE_PROPERTY"] >= p * 1000)
    )
]

# save prepared singledf, as csv and parquet
dfsingle.to_csv(
    rf"{Output_Eventset_Path}\NCEI-Storm_singlehazards\dfsingle_{inj}inj_{dth}dth_{c}c_{p}p_filtered_timelag{time_lag_int}_{start_year}-{end_year}.csv.gz",
    compression="gzip",
    encoding="utf-8",
    index=False,
)
dfsingle.to_parquet(
    rf"{Output_Eventset_Path}\NCEI-Storm_singlehazards\dfsingle_{inj}inj_{dth}dth_{c}c_{p}p_filtered_timelag{time_lag_int}_{start_year}-{end_year}.parquet.gzip",
    compression="gzip",
)

##CHECK WARNING####
pd.options.mode.chained_assignment = None  # default='warn'

# VERSION WITH TIME LAG CAPABILITY THAT ALSO CREATES PAIRS OF OVERLAPPING EVENTS

dfmulti = pd.DataFrame()

state_fips_list = dfsingle["STATE_FIPS"].unique().tolist()
state_fips_list.sort()

# Record any counties that don't have any overlapping hazard events for the defined time lag
No_Multihazard_County_df = pd.DataFrame()


# Function to check if datetime ranges overlap with a time lag
def datetime_ranges_overlap_with_lag(start1, end1, start2, end2, lag):
    return max(start1 - lag, start2 - lag) <= min(end1 + lag, end2 + lag)


def unique_pairs(pairs):
    unique_set = set()
    unique_list = []

    for pair in pairs:
        sorted_pair = tuple(sorted(pair))
        if sorted_pair not in unique_set:
            unique_set.add(sorted_pair)
            unique_list.append(pair)

    return unique_list


# Define a function to combine values in a column
def combine_values_comma(values):
    return ",".join(map(str, values))


def combine_values_slash(values):
    return "/".join(map(str, values))


def check_combine_values_comma(x):
    # Check if all values in the group are the same
    if (x == x.iloc[0]).all():
        return x.iloc[0]
    else:
        return ",".join(
            map(str, x)
        )  # If different, combine as a string with a delimiter


def check_combine_values_slash(x):
    # Check if all values in the group are the same
    if (x == x.iloc[0]).all():
        return x.iloc[0]
    else:
        return "/".join(
            map(str, x)
        )  # If different, combine as a string with a delimiter


def combine_multi(x):
    v1, v2 = x.iloc[0], x.iloc[1]
    v1_is_number = pd.to_numeric(v1, errors="coerce")
    v2_is_number = pd.to_numeric(v2, errors="coerce")

    if pd.notnull(v1_is_number) and pd.notnull(v2_is_number):
        return v1_is_number + v2_is_number
    elif pd.notnull(v1_is_number):
        return v1_is_number
    elif pd.notnull(v2_is_number):
        return v2_is_number
    else:
        return 0


pair_id_count = 0

for state_fips in tqdm(state_fips_list):
    print(f"state_fips:{state_fips}")
    state_df = dfsingle[dfsingle["STATE_FIPS"] == state_fips]
    county_fips_list = state_df["CZ_FIPS"].unique().tolist()
    county_fips_list.sort()

    for county_fips in tqdm(county_fips_list):
        print(f"state_fips:{state_fips}, county_fips:{county_fips}")

        df = state_df[state_df["CZ_FIPS"] == county_fips]

        # # Create DataFrame
        # df = pd.DataFrame(data)

        # Initialize the 'overlapping_events' column
        df.loc[:, "OVERLAPPING_EVENTS"] = [[] for _ in range(len(df))]

        # List to store pairs of events that overlap with different event types
        overlapping_event_pairs = []

        # Iterate over each row in the DataFrame

        for idx, row in df.iterrows():
            # Subset of rows with the same county location name and id
            subset = df[
                (df["CZ_FIPS"] == row["CZ_FIPS"])
                & (df["CZ_NAME"] == row["CZ_NAME"])
                & (df["STATE_FIPS"] == row["STATE_FIPS"])
            ]

            # Identify overlapping events with time lag
            for _, other_row in subset.iterrows():
                if row["EVENT_ID"] != other_row[
                    "EVENT_ID"
                ] and datetime_ranges_overlap_with_lag(
                    row["BEGIN_DATETIME"],
                    row["END_DATETIME"],
                    other_row["BEGIN_DATETIME"],
                    other_row["END_DATETIME"],
                    time_lag,
                ):
                    df.at[idx, "OVERLAPPING_EVENTS"].append((other_row["EVENT_ID"]))
                    if row["EVENT_TYPE"] != other_row["EVENT_TYPE"]:
                        overlapping_event_pairs.append(
                            ((row["EVENT_ID"]), (other_row["EVENT_ID"]))
                        )

        # Convert list to string for easier reading
        # df['OVERLAPPING_EVENTS'] = df['OVERLAPPING_EVENTS'].apply(lambda x: ', '.join(map(str, x)))
        df.loc[:, "OVERLAPPING_EVENTS"] = df["OVERLAPPING_EVENTS"].apply(
            lambda x: ",".join(map(str, x))
        )

        df = df[
            ["OVERLAPPING_EVENTS"]
            + [col for col in df.columns if col != "OVERLAPPING_EVENTS"]
        ]

        # Print the DataFrame to verify the result

        # print(df)
        # print("Pairs of overlapping events with different event types:")
        # print(overlapping_event_pairs)

        overlapping_event_pairs_unique = unique_pairs(overlapping_event_pairs)
        # print(len(overlapping_event_pairs_unique))

        all_pair_df = pd.DataFrame()

        # check to make sure there are some overlapping events, if not then skip to the next county iteration
        if len(overlapping_event_pairs_unique) > 0:

            for i in range(0, len(overlapping_event_pairs_unique)):
                pair_df_1 = df[df["EVENT_ID"] == overlapping_event_pairs_unique[i][0]][
                    0:1
                ]
                pair_df_1["BEGIN_LAT"] = pair_df_1["BEGIN_LAT"].round(2)
                pair_df_1["BEGIN_LON"] = pair_df_1["BEGIN_LON"].round(2)
                pair_df_1["END_LAT"] = pair_df_1["END_LAT"].round(2)
                pair_df_1["END_LON"] = pair_df_1["END_LON"].round(2)
                # pair_df_1 = pair_df_1.drop_duplicates() #remove any duplicates, there should only be 1 unique df row here
                # print(len(pair_df_1))

                pair_df_2 = df[df["EVENT_ID"] == overlapping_event_pairs_unique[i][1]][
                    0:1
                ]
                pair_df_2["BEGIN_LAT"] = pair_df_1["BEGIN_LAT"].round(2)
                pair_df_2["BEGIN_LON"] = pair_df_2["BEGIN_LON"].round(2)
                pair_df_2["END_LAT"] = pair_df_2["END_LAT"].round(2)
                pair_df_2["END_LON"] = pair_df_2["END_LON"].round(2)
                # pair_df_2 = pair_df_2.drop_duplicates() #remove any duplicates, there should only be 1 unique df row here
                # print(len(pair_df_1))

                pair_df_1["PAIR_ID"] = pair_id_count
                pair_df_2["PAIR_ID"] = pair_id_count

                # pair_df_1['PAIR_ID'] = i
                # pair_df_2['PAIR_ID'] = i

                pair_id_count = pair_id_count + 1

                all_pair_df = all_pair_df.drop_duplicates()
                temp_df = pd.concat([pair_df_1, pair_df_2])
                temp_df = temp_df.sort_values(
                    by="EVENT_TYPE"
                )  # reorder the two event pairs such that the event_type pairs are later formatted the same, when combined into a single string
                all_pair_df = pd.concat([all_pair_df, temp_df])

            all_pair_df = all_pair_df[
                ["PAIR_ID"] + [col for col in all_pair_df.columns if col != "PAIR_ID"]
            ]
            all_pair_df = all_pair_df.drop_duplicates()

            MULTI_INJURIES_DIRECT = all_pair_df.groupby("PAIR_ID")[
                "INJURIES_DIRECT"
            ].transform("sum")
            MULTI_INJURIES_INDIRECT = all_pair_df.groupby("PAIR_ID")[
                "INJURIES_INDIRECT"
            ].transform("sum")
            MULTI_DEATHS_DIRECT = all_pair_df.groupby("PAIR_ID")[
                "DEATHS_DIRECT"
            ].transform("sum")
            MULTI_DEATHS_INDIRECT = all_pair_df.groupby("PAIR_ID")[
                "DEATHS_INDIRECT"
            ].transform("sum")
            MULTI_DAMAGE_PROPERTY = all_pair_df.groupby("PAIR_ID")[
                "DAMAGE_PROPERTY"
            ].transform("sum")
            MULTI_DAMAGE_CROPS = all_pair_df.groupby("PAIR_ID")[
                "DAMAGE_CROPS"
            ].transform("sum")

            all_pair_df["MULTI_INJURIES_DIRECT"] = MULTI_INJURIES_DIRECT
            all_pair_df["MULTI_INJURIES_INDIRECT"] = MULTI_INJURIES_INDIRECT
            all_pair_df["MULTI_DEATHS_DIRECT"] = MULTI_DEATHS_DIRECT
            all_pair_df["MULTI_DEATHS_INDIRECT"] = MULTI_DEATHS_INDIRECT
            all_pair_df["MULTI_DAMAGE_PROPERTY"] = MULTI_DAMAGE_PROPERTY
            all_pair_df["MULTI_DAMAGE_CROPS"] = MULTI_DAMAGE_CROPS

            all_pair_df = all_pair_df.reindex(
                columns=[
                    "PAIR_ID",
                    "OVERLAPPING_EVENTS",
                    "EPISODE_ID",
                    "EVENT_ID",
                    "GEOID",
                    "STATE",
                    "STATE_FIPS",
                    "EVENT_TYPE",
                    "HAZARD",
                    "CZ_TYPE",
                    "CZ_FIPS",
                    "CZ_NAME",
                    "BEGIN_DATETIME",
                    "END_DATETIME",
                    "start_year",
                    "end_year",
                    "WFO",
                    "CZ_TIMEZONE",
                    "MULTI_INJURIES_DIRECT",
                    "INJURIES_DIRECT",
                    "MULTI_INJURIES_INDIRECT",
                    "INJURIES_INDIRECT",
                    "MULTI_DEATHS_DIRECT",
                    "DEATHS_DIRECT",
                    "MULTI_DEATHS_INDIRECT",
                    "DEATHS_INDIRECT",
                    "MULTI_DAMAGE_PROPERTY",
                    "DAMAGE_PROPERTY",
                    "MULTI_DAMAGE_CROPS",
                    "DAMAGE_CROPS",
                    "SOURCE",
                    "MAGNITUDE",
                    "MAGNITUDE_TYPE",
                    "FLOOD_CAUSE",
                    "CATEGORY",
                    "TOR_F_SCALE",
                    "TOR_LENGTH",
                    "TOR_WIDTH",
                    "TOR_OTHER_WFO",
                    "TOR_OTHER_CZ_STATE",
                    "TOR_OTHER_CZ_FIPS",
                    "TOR_OTHER_CZ_NAME",
                    "BEGIN_RANGE",
                    "BEGIN_AZIMUTH",
                    "BEGIN_LOCATION",
                    "END_RANGE",
                    "END_AZIMUTH",
                    "END_LOCATION",
                    "BEGIN_LAT",
                    "BEGIN_LON",
                    "END_LAT",
                    "END_LON",
                    "DATA_SOURCE",
                    "EPISODE_NARRATIVE",
                    "EVENT_NARRATIVE",
                ]
            )

            # input_df = input_df.reset_index(drop=True)
            # combined_df = input_df.groupby('PAIR_ID')

            # Group by 'Group' and aggregate
            # combined_df = input_df.groupby('PAIR_ID').agg(lambda x: x).reset_index()
            # combined_df = input_df.groupby('PAIR_ID',as_index=False).agg({
            #     'EPISODE_ID': combine_values_comma,
            #     'OVERLAPPING_EVENTS': combine_values_slash,
            #     'EVENT_ID': combine_values_comma,
            #'STATE': check_combine_values_comma,
            #     'STATE_FIPS': check_combine_values_comma,
            #     'EVENT_TYPE': combine_values_comma,
            #     'CZ_TYPE': check_combine_values_comma,
            #     'CZ_FIPS': check_combine_values_comma,
            #     'CZ_NAME': check_combine_values_comma,
            #     'WFO': combine_values_comma,
            #     'BEGIN_DATETIME': combine_values_comma,
            #     'END_DATETIME': combine_values_comma,
            #     'INJURIES_DIRECT': combine_values_comma,
            #     'MULTI_INJURIES_DIRECT': check_combine_values_comma,
            #     'INJURIES_INDIRECT': combine_values_comma,
            #     'MULTI_INJURIES_INDIRECT': check_combine_values_comma,
            #     'DEATHS_DIRECT': combine_values_comma,
            #     'MULTI_DEATHS_DIRECT': check_combine_values_comma,
            #     'DEATHS_INDIRECT': combine_values_comma,
            #     'MULTI_DEATHS_INDIRECT': check_combine_values_comma,
            #     'DAMAGE_PROPERTY': combine_values_comma,
            #     'MULTI_DAMAGE_PROPERTY': check_combine_values_comma,
            #     'DAMAGE_CROPS': combine_values_comma,
            #     'MULTI_DAMAGE_CROPS': check_combine_values_comma,
            #     'SOURCE': check_combine_values_comma,
            #     'MAGNITUDE': combine_values_comma,
            #     'MAGNITUDE_TYPE': combine_values_comma,
            #     'FLOOD_CAUSE': combine_values_comma,
            #     'CATEGORY': combine_values_comma,
            #     'BEGIN_LOCATION': combine_values_comma,
            #     'END_LOCATION': combine_values_comma,
            #     'BEGIN_LAT': combine_values_comma,
            #     'BEGIN_LON': combine_values_comma,
            #     'END_LAT': combine_values_comma,
            #     'END_LON': combine_values_comma,
            #     'EPISODE_NARRATIVE':check_combine_values_slash,
            #     'EVENT_NARRATIVE':check_combine_values_slash
            # }).reset_index()

            # combined_df = combined_df.reindex(columns=['PAIR_ID','OVERLAPPING_EVENTS','EPISODE_ID','EVENT_ID', 'GEOID', 'STATE','STATE_FIPS',
            #                                  'EVENT_TYPE','CZ_TYPE', 'CZ_FIPS', 'CZ_NAME', 'BEGIN_DATETIME', 'END_DATETIME', 'start_year', 'end_year',
            #                                  'WFO', 'CZ_TIMEZONE',
            #                                  'MULTI_INJURIES_DIRECT','INJURIES_DIRECT',
            #                                  'MULTI_INJURIES_INDIRECT','INJURIES_INDIRECT',
            #                                  'MULTI_DEATHS_DIRECT','DEATHS_DIRECT',
            #                                  'MULTI_DEATHS_INDIRECT','DEATHS_INDIRECT',
            #                                  'MULTI_DAMAGE_PROPERTY','DAMAGE_PROPERTY',
            #                                  'MULTI_DAMAGE_CROPS','DAMAGE_CROPS',
            #                                  'SOURCE','MAGNITUDE','MAGNITUDE_TYPE','FLOOD_CAUSE','CATEGORY','TOR_F_SCALE',
            #                                  'TOR_LENGTH','TOR_WIDTH', 'TOR_OTHER_WFO','TOR_OTHER_CZ_STATE',
            #                                  'TOR_OTHER_CZ_FIPS','TOR_OTHER_CZ_NAME','BEGIN_RANGE',
            #                                  'BEGIN_AZIMUTH','BEGIN_LOCATION','END_RANGE','END_AZIMUTH',
            #                                  'END_LOCATION','BEGIN_LAT','BEGIN_LON','END_LAT','END_LON','DATA_SOURCE','EPISODE_NARRATIVE', 'EVENT_NARRATIVE'])

            # all_combined_pair_df = pd.concat([all_combined_pair_df,combined_df])
            dfmulti = pd.concat([dfmulti, all_pair_df])

            # Print the combined DataFrame
        # else:
        #     No_Multihazard_County_df = pd.concat([No_Multihazard_County_df, pd.DataFrame({'STATE_FIPS': [state_fips], 'COUNTY_FIPS': [county_fips]})])
        #     print(f'\n NO MULTI-HAZARD PAIRS FOR STATE:{state_fips} AND COUNTY: {county_fips} \n')


print(f"Pair ID Count: {pair_id_count}")
display(dfmulti)

# save multidf, as csv and parquet
dfmulti.to_csv(
    rf"{Output_Eventset_Path}\NCEI-Storm_multihazards\dfmulti_{inj}inj_{dth}dth_{c}c_{p}p_filtered_timelag{time_lag_int}_{start_year}-{end_year}.csv.gz",
    compression="gzip",
    encoding="utf-8",
    index=False,
)
dfmulti.to_parquet(
    rf"{Output_Eventset_Path}\NCEI-Storm_multihazards\dfmulti_{inj}inj_{dth}dth_{c}c_{p}p_filtered_timelag{time_lag_int}_{start_year}-{end_year}.parquet.gzip",
    compression="gzip",
)
