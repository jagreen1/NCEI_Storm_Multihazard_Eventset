#######################
"""
Updated Nov, 2025

@author: Joshua Green - University of Southampton

Please cite this script/dataset if used in any research or publications.

Green, J. (2025) NCEI Storm Multihazard Eventset.
"""
#######################

import os, datetime
from datetime import datetime
import pandas as pd
import geopandas as gpd
import numpy as np
from tqdm import tqdm
import pickle

#pd.set_option('display.max_colwidth', None)
#pd.set_option('display.max_columns', None)

######################################################################################################
#                        USER DEFINED PARAMETERS
######################################################################################################
Cleaned_NCEI_Storm_Database_Parquet_Path = 'PATH TO CLEANED DATABASE PARQUET FILE'
Hazard_Eventset_Output_Path = 'PATH FOR OUTPUT FILES'
US_County_Shapefile_Path = 'PATH TO US CENSUS BUREAU COUNTY SHAPEFILE'
US_County_Shapefile_Path = 'https://github.com/jagreen1/NCEI_Storm_Multihazard_Eventset/raw/refs/heads/main/cb_2018_us_county_500k.shp'

# Define temporal year range
# CHANGE THESE VALUES AS DESIRED FOR TEMPORAL COVERAGE
start_year = 1996
end_year = 2024
year_range = range(start_year, end_year+2, 1)

# Define time lag in days
# CHANGE THESE VALUES AS DESIRED FOR APPROPRIATE TEMPORAL OVERLAP
#time_lag_days = 90
time_lag_days = 30
time_lag = pd.Timedelta(days=time_lag_days)
time_lag_int = time_lag.days

# Define which hazard event types to include in the database, see lookup table in comments below
# CHANGE THESE VALUES AS DESIRED FOR HAZARD/PERIL TYPE
#hazard_event_inclusion_filter = [ "av", "bz", "cfl", "cfl", "cw", "cw", "dd", "df", "dr", "ds", "ew", "ew", "ew", "fc", "ff", "ffg", "fg", "fl", "hl", "hs", "ht", "ht", "hw", "hw", "is", "les", "ls", "lt", "ltn", "mew", "mew", "mfg", "mhl", "mht", "mltn", "mtc", "mtps", "mtw", "nl", "p", "pfl", "rc", "se", "sl", "sm", "sn", "sst", "swv", "tc", "tc", "tn", "ts", "tw", "vo", "wf", "wp", "ws", "ww"]
hazard_event_inclusion_filter = ["av","bz","cfl","cw","df","dr","ds","ew","ff","fl","hl","ht","hw","is","les","ls","lt","ltn","p","pfl","se","sm","sn","sst","tc","tn","ts","tw","vo","wf","ws","ww"]
hazard_event_exclusion_filter = ["wp","fg","hs","fc","rc","dd","ffg","sl","nl","swv","mew","mtw","mhl","mht","mfg","mtc","mltn"]

# Impact filter thresholds, minimum values for including in final event set
# CHANGE THESE VALUES AS DESIRED FOR APPROPRIATE IMPACT FILTERING
inj = 1 # injuries
dth = 1 # deaths
c = 10  # crop damage in thousands
p = 10  # property damage in thousands

######################################################################################################
#                        MAIN SCRIPT
######################################################################################################
Hazard_Dict_Output_Path = rf'{Hazard_Eventset_Output_Path}\\Eventset_Dicts_{inj}inj_{dth}dth_{c}c_{p}p_lag{time_lag_days}_{start_year}-{end_year}'


def create_folder_if_not_exists(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"Folder created at: {folder_path}")
    else:
        print(f"Folder already exists at: {folder_path}")
create_folder_if_not_exists(Hazard_Eventset_Output_Path)
create_folder_if_not_exists(Hazard_Dict_Output_Path)


# Load the cleaned NCEI storm database
raw_df = pd.read_parquet(Cleaned_NCEI_Storm_Database_Parquet_Path)
# df['BEGIN_DATETIME'] = pd.to_datetime(df['BEGIN_DATETIME'])
# df['END_DATETIME'] = pd.to_datetime(df['END_DATETIME'])
dfevents = raw_df

dfevents["CZ_FIPS"] = dfevents["CZ_FIPS"].astype(str).str.zfill(3)
dfevents["STATE_FIPS"] = dfevents["STATE_FIPS"].astype(str).str.zfill(2)
dfevents["GEOID"] = dfevents["STATE_FIPS"].astype(str).str.zfill(2) + dfevents[
    "CZ_FIPS"
].astype(str).str.zfill(3)


# General qa/qc. Problems should have been removed during database cleaning, however complete additional final check.
dfevents = dfevents[~dfevents["EPISODE_ID"].isnull().isna()]
dfevents = dfevents[~dfevents["EVENT_ID"].isnull().isna()]
dfevents = dfevents[~dfevents["STATE"].isnull().isna()]
dfevents = dfevents[~dfevents["STATE_FIPS"].isnull().isna()]
dfevents = dfevents[~dfevents["EVENT_TYPE"].isnull().isna()]
dfevents = dfevents[~dfevents["CZ_FIPS"].isnull().isna()]
dfevents = dfevents[~dfevents["BEGIN_DATETIME"].isnull().isna()]
dfevents = dfevents[~dfevents["END_DATETIME"].isnull().isna()]

# Uncomment to remove these descriptive columns if desired
# dfevents = dfevents.drop(columns=['EPISODE_NARRATIVE', 'EVENT_NARRATIVE'])

# Define which hazard event types to include in the single/multi-hazard database
# "av":Avalanche,
# "bz":Blizzard,
# "cfl":Coastal Flood,
# "cw":Cold/Wind Chill,
# "dd":Dust Devil,
# "df":Debris Flow,
# "dr":Drought,
# "ds":Dust Storm,
# "ew":Extreme Wind,
# "fc":Funnel Cloud,
# "ff":Frost/Freeze,
# "ffg":Freezing Fog,
# "fg":Fog,
# "fl":Flood,
# "hl":Hail,
# "hs":High Surf,
# "ht":Hurricane/Typhoon,
# "hw":Heat,
# "is":Ice Storm,
# "les":Lake-Effect Snow,
# "ls":Landslide,
# "lt":Astronomical Low Tide,
# "ltn":Lightning,
# "mew":Marine Extreme Wind,
# "mfg":Marine Fog,
# "mhl":Marine Hail,
# "mht":Marine Hurricane/Typhoon,
# "mltn":Marine Lightning
# "mtc":Marine Tropical Storm/Depression,
# "mtw":Marine Thunderstorm Wind,
# "nl":Northern Lights,
# "p":Heavy Rain,
# "pfl":Rain Flood,
# "rc":Rip Current,
# "se":Seiche,
# "sl":Sleet,
# "sm":Dense Smoke,
# "sn":Snow,
# "sst":Storm Surge/Tide,
# "swv":Sneakerwave,
# "tc":Tropical Storm,
# "tn":Tornado,
# "ts":Tsunami,
# "tw":Thunderstorm Wind,
# "vo":Volcanic Ash,
# "wf":Wildfire,
# "wp":Waterspout,
# "ws":Winter Storm,
# "ww":Winter Weather,


# Define which events to include in the database
#hazard_event_inclusion_filter = [ "av", "bz", "cfl", "cfl", "cw", "cw", "dd", "df", "dr", "ds", "ew", "ew", "ew", "fc", "ff", "ffg", "fg", "fl", "hl", "hs", "ht", "ht", "hw", "hw", "is", "les", "ls", "lt", "ltn", "mew", "mew", "mfg", "mhl", "mht", "mltn", "mtc", "mtps", "mtw", "nl", "p", "pfl", "rc", "se", "sl", "sm", "sn", "sst", "swv", "tc", "tc", "tn", "ts", "tw", "vo", "wf", "wp", "ws", "ww"]

# Subset database to only desired event types
dfevents = dfevents[dfevents['HAZARD'].isin(hazard_event_inclusion_filter)]


# General qa/qc and preprocessing
dfevents['DEATHS_DIRECT'] = dfevents['DEATHS_DIRECT'].fillna(0).astype(int)
dfevents['DEATHS_INDIRECT'] = dfevents['DEATHS_INDIRECT'].fillna(0).astype(int)
dfevents['INJURIES_DIRECT'] = dfevents['INJURIES_DIRECT'].fillna(0).astype(int)
dfevents['INJURIES_DIRECT'] = dfevents['INJURIES_DIRECT'].fillna(0).astype(int)
dfevents["DAMAGE_CROPS"] = dfevents["DAMAGE_CROPS"].fillna(0).astype(int)
dfevents["DAMAGE_PROPERTY"] = dfevents["DAMAGE_PROPERTY"].fillna(0).astype(int)
dfevents['ADJ_DAMAGE_CROPS'] = dfevents['ADJ_DAMAGE_CROPS'].fillna(0).astype(int)
dfevents['ADJ_DAMAGE_PROPERTY'] = dfevents['ADJ_DAMAGE_PROPERTY'].fillna(0).astype(int)
dfevents['TOTAL_INJURIES'] = dfevents['TOTAL_INJURIES'].fillna(0)
dfevents['TOTAL_DEATHS'] = dfevents['TOTAL_DEATHS'].fillna(0)
dfevents['TOTAL_ADJ_DAMAGE'] = dfevents['TOTAL_ADJ_DAMAGE'].fillna(0)

dfevents.loc[dfevents['DEATHS_DIRECT']<0, 'DEATHS_DIRECT'] = 0
dfevents.loc[dfevents['DEATHS_INDIRECT']<0, 'DEATHS_INDIRECT'] = 0
dfevents.loc[dfevents['INJURIES_DIRECT']<0, 'INJURIES_DIRECT'] = 0
dfevents.loc[dfevents['INJURIES_DIRECT']<0, 'INJURIES_DIRECT'] = 0
dfevents.loc[dfevents['DAMAGE_CROPS']<0, 'DAMAGE_CROPS'] = 0
dfevents.loc[dfevents['DAMAGE_PROPERTY']<0, 'DAMAGE_PROPERTY'] = 0
dfevents.loc[dfevents['ADJ_DAMAGE_CROPS']<0, 'ADJ_DAMAGE_CROPS'] = 0
dfevents.loc[dfevents['ADJ_DAMAGE_PROPERTY']<0, 'ADJ_DAMAGE_PROPERTY'] = 0
dfevents.loc[dfevents['TOTAL_INJURIES']<0, 'TOTAL_INJURIES'] = 0
dfevents.loc[dfevents['TOTAL_DEATHS']<0, 'TOTAL_DEATHS'] = 0
dfevents.loc[dfevents['TOTAL_ADJ_DAMAGE']<0, 'TOTAL_ADJ_DAMAGE'] = 0

dfevents = dfevents.drop_duplicates()

# dfevents['start_year'] = dfevents['BEGIN_DATETIME'].dt.year
# dfevents['end_year'] = dfevents['END_DATETIME'].dt.year

dfevents = dfevents.reindex(
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
        "ADJ_DAMAGE_PROPERTY",
        "ADJ_DAMAGE_CROPS",
        'TOTAL_INJURIES',
        'TOTAL_DEATHS',
        'TOTAL_ADJ_DAMAGE',
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
dfevents = dfevents[
    (dfevents["BEGIN_DATETIME"].dt.year >= start_year)
    & (dfevents["END_DATETIME"].dt.year <= end_year)
]

# Remove unwanted state classes
# CHANGE THE EXCLUSED STATES AS DESIRED
# NOTE THAT THIS CLASSIFICATION INCLUDES US TERRITORIES AND WATER BODIES
Exclusion_State_List = [
    "ALASKA",
    "AMERICAN SAMOA",
    "ATLANTIC NORTH",
    "ATLANTIC SOUTH",
    "E PACIFIC",
    "GUAM WATERS",
    "GUAM",    
    "GULF OF ALASKA",
    "GULF OF MEXICO",
    "HAWAII WATERS",
    "HAWAII",
    "LAKE ERIE",
    "LAKE HURON",
    "LAKE MICHIGAN",
    "LAKE ONTARIO",
    "LAKE ST CLAIR",
    "LAKE SUPERIOR",
    "PUERTO RICO",
    "ST LAWRENCE R",
    "VIRGIN ISLANDS",
]
dfevents = dfevents[~dfevents["STATE"].isin(Exclusion_State_List)]


#remove marine zone only events, this should have been done as a byproduct of the above step, however this check is implemented as a backup
dfevents = dfevents[(dfevents['CZ_TYPE']!='M')]


# Filter by event impact
# Modify below to filter by 'ALL_INJURIES','ALL_DEATHS','TOTAL_ADJ_DAMAGE' if desired
dfevents = dfevents[
    (
        (dfevents["INJURIES_DIRECT"] >= inj)
        | (dfevents["INJURIES_INDIRECT"] >= inj)
        | (dfevents["DEATHS_DIRECT"] >= dth)
        | (dfevents["DEATHS_INDIRECT"] >= dth)
        | (dfevents["ADJ_DAMAGE_CROPS"] >= c * 1000)
        | (dfevents["ADJ_DAMAGE_PROPERTY"] >= p * 1000)
    )
]

# save prepared singledf, as a csv and/or parquet
# dfevents.to_csv(
#     rf"{Hazard_Eventset_Output_Path}\dfevents_{inj}inj_{dth}dth_{c}c_{p}p_lag{time_lag_int}_{start_year}-{end_year}.csv.gz",
#     compression="gzip",
#     encoding="utf-8",
#     index=False,
# )
dfevents.to_parquet(
    rf"{Hazard_Eventset_Output_Path}\dfevents_{inj}inj_{dth}dth_{c}c_{p}p_lag{time_lag_int}_{start_year}-{end_year}.parquet.gz",
    compression="gzip",
)

##CHECK WARNING####
pd.options.mode.chained_assignment = None  # default='warn'

# Define empty dataframes that will store the single hazard and multi-hazard eventsets
dfmulti = pd.DataFrame()
dfsingle = pd.DataFrame()

# Define a list of the state fips codes
state_fips_list = dfevents["STATE_FIPS"].unique().tolist()
state_fips_list.sort()

# Record any counties that don't have any overlapping hazard events for the defined time lag
#No_Multihazard_County_df = pd.DataFrame()


# Check if datetime ranges overlap with a time lag
def datetime_ranges_overlap_with_lag(start1, end1, start2, end2, lag):
    return max(start1 - lag, start2 - lag) <= min(end1 + lag, end2 + lag)


# Check to make sure that marine events can only be paired with other marine events
def cz_types_compatible(ct1, ct2):
    # both Z/C or both M
    return ((ct1 in ("Z", "C") and ct2 in ("Z", "C"))
            or (ct1 == "M" and ct2 == "M"))
    
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

# Set counter used to assign multihazard pair ids
pair_id_count = 0


for state_fips in tqdm(state_fips_list):
    
    all_combined_pair_df = pd.DataFrame()
    
    print(f"state_fips:{state_fips}")
    state_df = dfevents[dfevents["STATE_FIPS"] == state_fips]
    county_fips_list = sorted(state_df["CZ_FIPS"].unique().tolist())

    for county_fips in tqdm(county_fips_list):
        print(f"state_fips:{state_fips}, county_fips:{county_fips}")

        df = state_df[state_df["CZ_FIPS"] == county_fips]

        # # Create DataFrame
        # df = pd.DataFrame(data)

        # Initialize the 'overlapping_events' column
        #df.loc[:, "OVERLAPPING_EVENTS"] = [[] for _ in range(len(df))]
        df["OVERLAPPING_EVENTS"] = [[] for _ in range(len(df))]

        # List to store pairs of events that overlap with different event types
        overlapping_event_pairs = []

        # # Iterate over each row in the DataFrame
        # for idx, row in df.iterrows():
        #     # Subset of rows with the same county location name and id
        #     subset = df[
        #         (df["CZ_FIPS"] == row["CZ_FIPS"])
        #         & (df["CZ_NAME"] == row["CZ_NAME"])
        #         & (df["STATE_FIPS"] == row["STATE_FIPS"])
        #     ]

        #     # Identify overlapping events with time lag
        #     for _, other_row in subset.iterrows():
        #         # Check to make sure that the event is not paired with itself, i.e. the event ids are different
        #         # Check that events overlap in time
        #         # Check
        #         if (row["EVENT_ID"] != other_row["EVENT_ID"] 
        #         and datetime_ranges_overlap_with_lag( 
        #             row["BEGIN_DATETIME"],
        #             row["END_DATETIME"],
        #             other_row["BEGIN_DATETIME"],
        #             other_row["END_DATETIME"],
        #             time_lag)
        #         and (cz_types_compatible(
        #             row["CZ_TYPE"],
        #             other["CZ_TYPE"])
        #             ):
                    
        #             df.at[idx, "OVERLAPPING_EVENTS"].append((other_row["EVENT_ID"])) 
        #             # Check to see if events are not the same type, excluding the pairing of the like-type events, 
        #             # Some events are recorded in parts despite being the same event, this avoids paring an event with itself
        #             if row["EVENT_TYPE"] != other_row["EVENT_TYPE"]:
        #                 overlapping_event_pairs.append(
        #                     ((row["EVENT_ID"]), (other_row["EVENT_ID"]))
        #                 )
        
        # Iterate over each row in the DataFrame
        for idx, row in df.iterrows():
            # Subset of rows with the same county location name and id
            subset = df[
                (df["CZ_FIPS"] == row["CZ_FIPS"]) &
                (df["CZ_NAME"] == row["CZ_NAME"]) &
                (df["STATE_FIPS"] == row["STATE_FIPS"])
            ]
            for _, other in subset.iterrows():
                #Check to make sure that the event is not paired with itself, if true skip iteration
                # Note that events with the same EPISODE_ID (i.e. storm episode) are allowed, just not the same individual storm event
                if row["EVENT_ID"] == other["EVENT_ID"]:
                    continue

                # Check if there are temporally overlapping events, if false skip iteration
                if not datetime_ranges_overlap_with_lag(
                        row["BEGIN_DATETIME"], row["END_DATETIME"],
                        other["BEGIN_DATETIME"], other["END_DATETIME"],
                        time_lag):
                    continue

                # UNCOMMENT IF DESIRED
                # Check if the overlapping events satisfy the CZ_TYPE pair rules, 'M' can only be paired with 'M', if false skip iteration
                #if not cz_types_compatible(row["CZ_TYPE"], other["CZ_TYPE"]):
                #    continue

                df.at[idx, "OVERLAPPING_EVENTS"].append(other["EVENT_ID"])

                # Check if the hazard event types are different (to avoid self-duplication), if true add event pairs to pair dataframe
                if row["EVENT_TYPE"] != other["EVENT_TYPE"]:
                    
                    overlapping_event_pairs.append(((row["EVENT_ID"]), (other["EVENT_ID"])))

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

        # Check to make sure there are some overlapping events, if not then skip to the next county iteration
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
                pair_df_2["BEGIN_LAT"] = pair_df_2["BEGIN_LAT"].round(2)
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

                temp_df = pd.concat([pair_df_1, pair_df_2])
                temp_df = temp_df.sort_values(
                    by="EVENT_TYPE"
                )  # Reorder the two event pairs such that the event_type pairs are later formatted the same, when combined into a single string
                all_pair_df = pd.concat([all_pair_df, temp_df])
                all_pair_df = all_pair_df.drop_duplicates()

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
                "ADJ_DAMAGE_PROPERTY"
            ].transform("sum")
            MULTI_DAMAGE_CROPS = all_pair_df.groupby("PAIR_ID")[
                "ADJ_DAMAGE_CROPS"
            ].transform("sum")

            all_pair_df["MULTI_INJURIES_DIRECT"] = MULTI_INJURIES_DIRECT
            all_pair_df["MULTI_INJURIES_INDIRECT"] = MULTI_INJURIES_INDIRECT
            all_pair_df["MULTI_DEATHS_DIRECT"] = MULTI_DEATHS_DIRECT
            all_pair_df["MULTI_DEATHS_INDIRECT"] = MULTI_DEATHS_INDIRECT
            all_pair_df["MULTI_ADJ_DAMAGE_PROPERTY"] = MULTI_DAMAGE_PROPERTY
            all_pair_df["MULTI_ADJ_DAMAGE_CROPS"] = MULTI_DAMAGE_CROPS

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
                    "MULTI_ADJ_DAMAGE_PROPERTY",
                    "ADJ_DAMAGE_PROPERTY",
                    "MULTI_ADJ_DAMAGE_CROPS",
                    "ADJ_DAMAGE_CROPS",
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

            #all_combined_pair_df = pd.concat([all_combined_pair_df,combined_df])
            all_combined_pair_df = pd.concat([all_combined_pair_df,all_pair_df])
            dfmulti = pd.concat([dfmulti, all_pair_df])


            # Print the combined DataFrame
        # else:
        #     No_Multihazard_County_df = pd.concat([No_Multihazard_County_df, pd.DataFrame({'STATE_FIPS': [state_fips], 'COUNTY_FIPS': [county_fips]})])
        #     print(f'\n NO MULTI-HAZARD PAIRS FOR STATE:{state_fips} AND COUNTY: {county_fips} \n')

    #save the multihazard df for each state as an individual file, these can then be combined afterwards
    #this can be used to avoids having a df in memory with all multihazard events for the entire US, which could result in memory errors
    #all_combined_pair_df.to_csv(fr'{Hazard_Eventset_Output_Path}/NCEI_Storm_Database_Multihazards_1996_2024_lag_{time_lag_int}_state_{state_fips}.csv.gz', compression='gzip', encoding='utf-8', index=True)


# print(f"Pair ID Count: {pair_id_count}")
# display(dfmulti)

# save multidf, as a csv and/or parquet
# dfmulti.to_csv(
#     rf"{Hazard_Eventset_Output_Path}/dfmulti_{inj}inj_{dth}dth_{c}c_{p}p_lag{time_lag_int}_{start_year}-{end_year}.csv.gz",
#     compression="gzip",
#     encoding="utf-8",
#     index=False,
# )


# Load us county shapefile, used to complete spatial filtering, can implement via shapely.STRtree() 
us_county_polygons = gpd.read_file(US_County_Shapefile_Path)
us_county_polygons = us_county_polygons.dissolve(by='GEOID')# some of the counties have multiple small polygons, dissolve into one single polygon
us_county_polygons = us_county_polygons.reset_index(drop=False)
#us_county_polygons['GEOID'] = us_county_polygons['GEOID'].astype(int).astype(str) #remove leading zeros
us_county_polygons['GEOID'] = us_county_polygons['GEOID'].astype(str).str.zfill(5) #add leading zeros

print(f'US County Polygon CRS: {us_county_polygons.geometry.crs}')

# Define dictionaries that will store county event info, in a 3x nested structure of year->state->county
single_hazard_count_dict = {}
single_hazard_event_dict = {}

multihazard_count_dict = {}
multihazard_event_dict = {}

no_hazard_boolean_dict = {}
single_hazard_boolean_dict = {}
multihazard_boolean_dict = {}
no_hazard_or_single_hazard_boolean_dict = {}
single_hazard_or_multihazard_boolean_dict = {}

# Define list of states to iterate through
state_list = us_county_polygons['STATEFP'].unique().tolist() 


# Iterate through the previous defined start/end years
for year in tqdm(year_range):
    print(f'Year: {year}')
    # Define nested structure of dictionaries
    single_hazard_count_dict[year] = {}
    multihazard_count_dict[year] = {}
    single_hazard_event_dict[year] = {}
    multihazard_event_dict[year] = {}
    no_hazard_boolean_dict[year] = {}
    single_hazard_boolean_dict[year] = {}
    multihazard_boolean_dict[year] = {}
    no_hazard_or_single_hazard_boolean_dict[year] = {}
    single_hazard_or_multihazard_boolean_dict[year] = {}
    
    dfsingle_sub = dfevents[(dfevents['start_year']==year) | (dfevents['end_year']==year)].reset_index(drop=True)
    dfmulti_sub = dfmulti[(dfmulti['start_year']==year) | (dfmulti['end_year']==year)].reset_index(drop=True)

    for state in state_list:
        #print(f'State: {state}')
        # Define nested structure of dictionaries
        single_hazard_count_dict[year][state] = {}
        multihazard_count_dict[year][state] = {}
        single_hazard_event_dict[year][state] = {}
        multihazard_event_dict[year][state] = {}
        no_hazard_boolean_dict[year][state] = {}
        single_hazard_boolean_dict[year][state] = {}
        multihazard_boolean_dict[year][state] = {}
        no_hazard_or_single_hazard_boolean_dict[year][state] = {}
        single_hazard_or_multihazard_boolean_dict[year][state] = {}
        
        county_state_list = us_county_polygons.loc[us_county_polygons['STATEFP'] == state, ['GEOID','COUNTYFP','geometry']]

        # Currently using geoid to index, could user countyfp instead, would have to change some things
        for county in county_state_list['GEOID']:
            #print(county)
            
            # SPATIAL GEOMETRY FILTER APPROACH
            ###############################################################
            # county_geom = us_county_polygons.loc[us_county_polygons['GEOID'] == str(county), 'geometry'].iloc[0]
            # #single spatial filter
            # tree = shapely.STRtree(dfsingle_sub.Geometry.values) # Make tree too see Geometry overlap
            # arr1 = np.transpose(tree.query(county_geom, predicate='intersects'))  # Find intersecting hazards with the area of interest
            # # crop hazard data to relevant regions 
            # dfsingle_sub_county = dfsingle_sub.loc[np.sort(arr1)].reset_index(drop=True) # Remove the hazards that do not intersect with the area of interest	
            # # multi spatial filter
            # tree2 = shapely.STRtree(dfmulti_sub.Geometry.values) # Make tree too see Geometry overlap
            # arr2 = np.transpose(tree2.query(county_geom, predicate='intersects'))  # Find intersecting hazards with the area of interest
            # # crop hazard data to relevant regions 
            # dfmulti_sub_county = dfmulti_sub.loc[np.sort(arr2)].reset_index(drop=True) # Remove the hazards that do not intersect with the area of interest	
            ###############################################################
            
            # NON GEOMETRY SPATIAL FILTER APPROACH, FILTER VIA COUNTY GEOID
            ###############################################################
            dfsingle_sub_county = dfsingle_sub[dfsingle_sub['GEOID']==county].reset_index(drop=True) # Remove the hazards that do not intersect with the area of interest	
            dfmulti_sub_county = dfmulti_sub[dfmulti_sub['GEOID']==county].reset_index(drop=True) # Remove the hazards that do not intersect with the area of interest	
            ###############################################################

            
            # Add single to dict
            # Check if there are matching single events in the multi event, this will be true if multi is true, then remove the multi events from the single before adding to dict
            multi_events = set(dfmulti_sub_county['EVENT_ID'].unique())
            single_events = set(dfsingle_sub_county['EVENT_ID'].unique())
            single_only_events = single_events.symmetric_difference(multi_events) #find the 'code' ids of events in only single
            
            # Remove the single hazards that make up a valid multihazard for that location, so its single hazard only events
            dfsingle_only_sub_county = dfsingle_sub_county[dfsingle_sub_county['EVENT_ID'].isin(single_only_events)]
            
            # If single hazard, add the count of unique single hazard events to the dict
            # Could alternatively use the 'id' as a unique field, I believe both 'code' and 'id' are unique for the single hazards
            if len(dfsingle_only_sub_county)>0:
                single_hazard_boolean_dict[year][state][county] = True
                single_hazard_count_dict[year][state][county] = len(dfsingle_only_sub_county['EVENT_ID'].unique())
                single_hazard_event_dict[year][state][county] = dfsingle_only_sub_county['EVENT_ID'].unique().tolist()
            else:
                single_hazard_boolean_dict[year][state][county] = False
                single_hazard_count_dict[year][state][county] = 0
                single_hazard_event_dict[year][state][county] = []

            # Add multi to dict
            multi_duplicated_events = dfmulti_sub_county["PAIR_ID"][dfmulti_sub_county["PAIR_ID"].duplicated(keep=False)]
            multi_filtered_df = dfmulti_sub_county[dfmulti_sub_county["PAIR_ID"].isin(multi_duplicated_events)]    
            # If multihazard, add the count of unique multihazard events to dict
            # If len(dfmulti_sub_county['code'].unique().tolist())>0:
            if len(multi_filtered_df)>0:
                multihazard_boolean_dict[year][state][county] = True
                multihazard_count_dict[year][state][county] = len(multi_filtered_df['PAIR_ID'].unique())
                multihazard_event_dict[year][state][county] = multi_filtered_df['PAIR_ID'].unique().tolist()
                #OR could add the 'code' id of the single hazard events that make up a multihazard
                #multihazard_event_dict[year][state][county] = {multi_filtered_df['code'].unique().tolist()}
            else:
                multihazard_boolean_dict[year][state][county] = False
                multihazard_count_dict[year][state][county] = 0
                multihazard_event_dict[year][state][county] = []
            
            # Add no hazard to dict
            if ((len(dfsingle_only_sub_county)==0) & (len(multi_filtered_df)==0)):
                no_hazard_boolean_dict[year][state][county] = True
            else:
                no_hazard_boolean_dict[year][state][county] = False

            # Add no single only hazard to dict, could remove the first two conditions,
            if (((len(dfsingle_only_sub_county)==0) | (len(dfsingle_only_sub_county)>0)) & (len(multi_filtered_df)==0)):
                no_hazard_or_single_hazard_boolean_dict[year][state][county] = True
            else:
                no_hazard_or_single_hazard_boolean_dict[year][state][county] = False

            # Add single hazard or multi-hazard (i.e. inverse of no hazard) to dict
            if ((len(dfsingle_only_sub_county)>0) | (len(multi_filtered_df)>0)):
                single_hazard_or_multihazard_boolean_dict[year][state][county] = True
            else:
                single_hazard_or_multihazard_boolean_dict[year][state][county] = False


#Save final dictionaries as pickle

with open(Hazard_Dict_Output_Path+f'\\NCEI_County_SH_only_event_dict.pkl', 'wb') as file:
    pickle.dump(single_hazard_event_dict, file)

with open(Hazard_Dict_Output_Path+f'\\NCEI_County_MH_event_dict.pkl', 'wb') as file:
    pickle.dump(multihazard_event_dict, file)

with open(Hazard_Dict_Output_Path+f'\\NCEI_County_MH_count_dict.pkl', 'wb') as file:
    pickle.dump(multihazard_count_dict, file)

with open(Hazard_Dict_Output_Path+f'\\NCEI_County_SH_count_dict.pkl', 'wb') as file:
    pickle.dump(single_hazard_count_dict, file)

with open(Hazard_Dict_Output_Path+f'\\NCEI_County_NH_boolean_dict.pkl', 'wb') as file:
    pickle.dump(no_hazard_boolean_dict, file)
    
with open(Hazard_Dict_Output_Path+f'\\NCEI_County_SH_boolean_dict.pkl', 'wb') as file:
    pickle.dump(single_hazard_boolean_dict, file)

with open(Hazard_Dict_Output_Path+f'\\NCEI_County_MH_boolean_dict.pkl', 'wb') as file:
    pickle.dump(multihazard_boolean_dict, file)

with open(Hazard_Dict_Output_Path+f'\\NCEI_County_SH_NH_boolean_dict.pkl', 'wb') as file:
    pickle.dump(no_hazard_or_single_hazard_boolean_dict, file)

with open(Hazard_Dict_Output_Path+f'\\NCEI_County_SH_MH_boolean_dict.pkl', 'wb') as file:
    pickle.dump(single_hazard_or_multihazard_boolean_dict, file)

print("All hazard dicts saved as pickle")


# Function to access the values from the x3 nested dictionaries
def get_values(data):
    values = []
    for value in data.values():
        if isinstance(value, dict):  # Check if the value is another dictionary
            values.extend(get_values(value))  # Recursively collect values
        else:
            values.extend(value)  # Add the list of values directly
    return values


# # Load single-only hazard event dict
# with open(Hazard_Dict_Output_Path+f'\\NCEI_County_SH_only_event_dict_{start_year}-{end_year}.pkl', 'rb') as file:
#     single_hazard_event_dict = pickle.load(file)

# Subset single hazard events to single-only hazard (single hazards that do not make up a multi-hazard pair)
single_only_hazard_events = get_values(single_hazard_event_dict)
dfsingle = dfevents[dfevents['EVENT_ID'].isin(single_only_hazard_events)]

dfsingle.to_parquet(
    rf"{Hazard_Eventset_Output_Path}/dfsingle_{inj}inj_{dth}dth_{c}c_{p}p_lag{time_lag_int}_{start_year}-{end_year}.parquet.gz",
    compression="gzip",
)

dfmulti.to_parquet(
    rf"{Hazard_Eventset_Output_Path}/dfmulti_{inj}inj_{dth}dth_{c}c_{p}p_lag{time_lag_int}_{start_year}-{end_year}.parquet.gz",
    compression="gzip",
)

print(f'Total Number of Hazard Events: {len(dfevents)}')
print(f'Number of Single Hazard Only Events:{len(dfsingle)}')
print(f'Number of Multi-Hazard Events:{int(len(dfmulti)/2)}')
