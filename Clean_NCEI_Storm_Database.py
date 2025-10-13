#######################
"""
Created March 21st, 2025

@author: Joshua Green - University of Southampton

Please cite this dataset if used in any publications.

Green, J. (2025) NCEI Storm Multihazard Eventset.
"""
#######################


from datetime import datetime
import functools
import glob
import numpy as np
import pandas as pd
import os
import re
from tqdm import tqdm

"""
This is a directory containing annual csv files from 1950 to 2024+
The input database files necessary to run these scripts can be downloaded via HTML/FTP on the NCEI website at the below URLs (as of Mar 2025).
- https://www.ncdc.noaa.gov/stormevents/ftp.jsp
- https://www.ncei.noaa.gov/pub/data/swdi/stormevents/csvfiles/
- ftp://ftp.ncei.noaa.gov/pub/data/swdi/stormevents/csvfiles/
"""
NCEI_Storm_Database_Bulk_FTP_Download_Path = r"PATH GOES HERE"
Output_Cleaned_Database_Path = r"PATH GOES HERE"


base_dir = r"{NCEI_Storm_Database_Bulk_FTP_Download_Path}"

NWS_Z_to_CZ_Fips_df = pd.read_csv(
    r"https://github.com/jagreen1/NCEI_Storm_Multihazard_Eventset/blob/main/NWS_Zone_to_County_FIPS_bp18mr25.dbx.txt",
    delimiter="|",
)


def load_files(pattern):
    files = f"{base_dir}/{pattern}"
    csv_files = glob.glob(
        os.path.join(base_dir, "StormEvents_details-ftp_v1.0_*.csv.gz")
    )
    dataframes = [pd.read_csv(x, low_memory=False) for x in csv_files]
    return pd.concat(dataframes).reset_index(drop=True)


df_details = load_files("*details*.csv")

df_details = df_details[~df_details["EPISODE_ID"].isnull()]
df_details["EPISODE_ID"] = df_details["EPISODE_ID"].astype(int)
df_details = df_details[~df_details["EVENT_ID"].isnull()]
df_details["EVENT_ID"] = df_details["EVENT_ID"].astype(int)
df_details = df_details[~df_details["STATE"].isnull()]
df_details = df_details[~df_details["STATE_FIPS"].isnull()]
df_details["STATE_FIPS"] = df_details["STATE_FIPS"].astype(int)
df_details = df_details[~df_details["EVENT_TYPE"].isnull()]
df_details = df_details[~df_details["CZ_FIPS"].isnull()]

df_details = df_details.drop_duplicates()


# convert the NWS zones in CZ_FIPS (for CZ_TYPE = Z) to actual CZ FIPS values

# Filter the rows where CZ_TYPE equals 'C'
cz_type_c = df_details[df_details["CZ_TYPE"] == "C"]

# Create a mapping of (STATE_FIPS, STATE, CZ_NAME) to CZ_FIPS for rows with CZ_TYPE = 'C'
mapping = cz_type_c.set_index(["STATE_FIPS", "STATE", "CZ_NAME"])["CZ_FIPS"].to_dict()

# Function to replace CZ_FIPS based on the mapping
def replace_cz_fips(row):
    if row["CZ_TYPE"] == "Z":
        key = (row["STATE_FIPS"], row["STATE"], row["CZ_NAME"])
        return mapping.get(
            key, row["CZ_FIPS"]
        )  # Replace if a match is found; otherwise keep original
    return row["CZ_FIPS"]


# Apply the function to update the CZ_FIPS column
df_details["CZ_FIPS"] = df_details.apply(replace_cz_fips, axis=1)


df_details["CZ_FIPS"] = df_details["CZ_FIPS"].astype(str).str.zfill(3)
df_details["STATE_FIPS"] = df_details["STATE_FIPS"].astype(str).str.zfill(2)
df_details["GEOID"] = df_details["STATE_FIPS"].astype(str).str.zfill(2) + df_details[
    "CZ_FIPS"
].astype(str).str.zfill(3)

df_details.SOURCE = df_details.SOURCE.str.title()

ACRONYMS = ["Asos", "Awos", "Awss", "Nws", "C-Man", "Raws", "Shave", "Snotel", "Wlon"]
for acronym in ACRONYMS:
    pattern = f"\\b{acronym}\\b"
    replacement = acronym.upper()
    df_details.SOURCE = df_details.SOURCE.str.replace(pattern, replacement, regex=True)

source_substitutions = {
    "Arpt Equip(AWOS,ASOS)": "AWOS,ASOS,Mesonet,Etc",
    "Coastal Observing Station": "Coast Guard",
    "Cocorahs": "CoCoRaHS",
    "Coop Observer": "Cooperative Network Observer",
    "Coop Station": "Cooperative Network Observer",
    "Dept Of Highways": "Department Of Highways",
    "Fire Dept/Rescue Squad": "Fire Department/Rescue",
    "General Public": "Public",
    "Govt Official": "State Official",
    "Manual Input": "Unknown",
    "Meteorologist(Non NWS)": "Public",
    "NWS Employee(Off Duty)": "NWS Employee",
    "Npop": "Unknown",
    "Official NWS Obs.": "Official NWS Observations",
}

for original, replacement in source_substitutions.items():
    df_details.SOURCE = df_details.SOURCE.str.replace(original, replacement)


# standardize hazard event names

event_substitution = {
    r"^HAIL.*": "Hail",
    r"^High Snow$": "Heavy Snow",
    r"^Hurricane$": "Hurricane",
    r"^OTHER$": "Dust Devil",
    r"^THUNDERSTORM WIND.*": "Thunderstorm Wind",
    r"^TORNADO.*": "Tornado",
    r"^Volcanic Ashfall.*$": "Volcanic Ash",
}


for original, replacement in event_substitution.items():
    df_details.EVENT_TYPE = df_details.EVENT_TYPE.str.replace(
        original, replacement, regex=True
    )

df_details["EVENT_TYPE"] = df_details["EVENT_TYPE"].str.replace(
    "TropicalDepression", "Tropical Depression", regex=False
)
df_details["EVENT_TYPE"] = df_details["EVENT_TYPE"].str.replace(
    "Hurricane (Typhoon)", "Hurricane/Typhoon", regex=False
)

# map HAZARD col using acronym dict
acronym_map = {
    "Heavy Snow": "sn",
    "High Wind": "ew",
    "Winter Storm": "ws",
    "Tornado": "tn",
    "Lightning": "ltn",
    "Hail": "hl",
    "Flood": "fl",
    "Thunderstorm Wind": "tw",
    "Ice Storm": "is",
    "Waterspout": "wp",
    "Winter Weather": "ww",
    "Coastal Flood": "cfl",
    "Cold/Wind Chill": "cw",
    "Dense Fog": "fg",
    "Avalanche": "av",
    "Blizzard": "bz",
    "Frost/Freeze": "ff",
    "Flash Flood": "pfl",
    "High Surf": "hs",
    "Heavy Rain": "p",
    "Dust Storm": "ds",
    "Heat": "hw",
    "Funnel Cloud": "fc",
    "Drought": "dr",
    "Debris Flow": "df",
    "Wildfire": "wf",
    "Strong Wind": "ew",
    "Dust Devil": "dd",
    "Rip Current": "rc",
    "Tropical Storm": "tc",
    "Hurricane/Typhoon": "ht",
    "Storm Surge/Tide": "sst",
    "Freezing Fog": "ffg",
    "Marine High Wind": "mew",
    "Sleet": "sl",
    "Lake-Effect Snow": "les",
    "Astronomical Low Tide": "lt",
    "Volcanic Ash": "vo",
    "Seiche": "se",
    "Extreme Cold/Wind Chill": "cw",
    "Excessive Heat": "hw",
    "Heavy Wind": "ew",
    "Marine Thunderstorm Wind": "mtw",
    "Northern Lights": "nl",
    "Marine Hail": "mhl",
    "Dense Smoke": "sm",
    "Tsunami": "ts",
    "Landslide": "ls",
    "Marine Strong Wind": "mew",
    "Lakeshore Flood": "cfl",
    "Tropical Depression": "tc",
    "Marine Hurricane/Typhoon": "mht",
    "Marine Dense Fog": "mfg",
    "Marine Tropical Storm": "mtps",
    "Sneakerwave": "swv",
    "Marine Lightning": "mltn",
    "Marine Tropical Depression": "mtc",
}
df_details["HAZARD"] = df_details["EVENT_TYPE"].map(acronym_map)

# NOTE: There are several similar hazards that I have chosen to group together, see below:
# ew - 'High Wind' & 'Strong Wind'
# cw - 'Cold/Wind Chill' & 'Extreme Cold/Wind Chill'
# hw - 'Heat' & 'Excessive Heat'
# mew - 'Marine High Wind' & 'Marine Strong Wind'
# tc - 'Tropical Storm' & 'TropicalDepression'
# mtc - 'Marine Tropical Storm' & 'Marine Tropical Depression'
# cfl - 'Coastal Flood' & 'Lakeshore Flood'


# standardize timezones

timezone_substitutions = {
    "CDT": "CST",
    "CSC": "CST",  # event was in Iowa
    "EDT": "EST",
    "GMT": "CST",  # event was in Louisiana
    "GST": "ChST",  # events were in Guam, which uses Chamorro Standard Time
    "MDT": "MST",
    "PDT": "PST",
    "SCT": "CST",  # event was in Wisconsin
}
unknown_timezones = {
    "HAWAII": "HST",
    "OKLAHOMA": "CST",
    "MASSACHUSETTS": "EST",
    "GEORGIA": "EST",
    "ILLINOIS": "CST",
}

df_details.CZ_TIMEZONE = df_details.CZ_TIMEZONE.str.replace(
    r"-*\d*$", "", regex=True
).str.upper()

for original, replacement in timezone_substitutions.items():
    df_details.CZ_TIMEZONE = df_details.CZ_TIMEZONE.str.replace(original, replacement)

for index, row in df_details.query('CZ_TIMEZONE=="UNK"').iterrows():
    df_details.at[index, "CZ_TIMEZONE"] = unknown_timezones.get(row.STATE, "UNK")


def create_datetime(df, prefix):
    df_components = pd.to_datetime(
        {
            "year": df[f"{prefix}YEARMONTH"] // 100,
            "month": df[f"{prefix}YEARMONTH"] % 100,
            "day": df[f"{prefix}DAY"],
            "hour": df[f"{prefix}TIME"] // 100,
            "minute": df[f"{prefix}TIME"] % 100,
        }
    )
    return pd.to_datetime(df_components)


legacy = ["BEGIN_YEARMONTH", "BEGIN_DAY", "BEGIN_TIME", "BEGIN_DATE_TIME"]
legacy = legacy + ["END_YEARMONTH", "END_DAY", "END_TIME", "END_DATE_TIME"]
legacy = legacy + ["MONTH_NAME", "YEAR"]

df_details["BEGIN_DATETIME"] = create_datetime(df_details, "BEGIN_")
df_details["END_DATETIME"] = create_datetime(df_details, "END_")
df_details = df_details.drop(columns=legacy)

df_details["start_year"] = df_details["BEGIN_DATETIME"].dt.year
df_details["end_year"] = df_details["END_DATETIME"].dt.year

# standardize costs
def to_cost(column):
    price = column[column.notnull()].astype("str").str.upper()

    valid_price = r"^[\d.]+[KMB]?$"
    price = price[price.str.contains(valid_price, regex=True)]
    has_K = price.str.contains("K")
    has_M = price.str.contains("M")
    has_B = price.str.contains("B")
    price = price.str.replace(r"[KMB]", "", regex=True).astype("float")

    scale = np.select([has_K, has_M, has_B], [1000, 1_000_000, 1_000_000_000], 1)
    return scale * price

#standardize damage values
df_details.DAMAGE_PROPERTY = to_cost( df_details.DAMAGE_PROPERTY)
df_details.DAMAGE_CROPS = to_cost( df_details.DAMAGE_CROPS)


#fix invalid values
df_details['DEATHS_DIRECT'] = df_details['DEATHS_DIRECT'].fillna(0).astype(int)
df_details['DEATHS_INDIRECT'] = df_details['DEATHS_INDIRECT'].fillna(0).astype(int)
df_details['INJURIES_DIRECT'] = df_details['INJURIES_DIRECT'].fillna(0).astype(int)
df_details['INJURIES_DIRECT'] = df_details['INJURIES_DIRECT'].fillna(0).astype(int)
df_details['DAMAGE_CROPS'] = df_details['DAMAGE_CROPS'].fillna(0).astype(int)
df_details['DAMAGE_PROPERTY'] = df_details['DAMAGE_PROPERTY'].fillna(0).astype(int)
df_details.loc[df_details['DEATHS_DIRECT']<0, 'DEATHS_DIRECT'] = 0
df_details.loc[df_details['DEATHS_INDIRECT']<0, 'DEATHS_INDIRECT'] = 0
df_details.loc[df_details['INJURIES_DIRECT']<0, 'INJURIES_DIRECT'] = 0
df_details.loc[df_details['INJURIES_DIRECT']<0, 'INJURIES_DIRECT'] = 0
df_details.loc[df_details['DAMAGE_PROPERTY']<0, 'DAMAGE_PROPERTY'] = 0
df_details.loc[df_details['DAMAGE_CROPS']<0, 'DAMAGE_CROPS'] = 0

#add new combined impact fields
dfsingle['ALL_DAMAGE'] = (dfsingle['DAMAGE_PROPERTY'] + dfsingle['DAMAGE_CROPS']).fillna(0)
dfsingle['ALL_DEATHS'] = (dfsingle['DEATHS_DIRECT'] + dfsingle['DEATHS_INDIRECT']).fillna(0)
dfsingle['ALL_INJURIES'] = (dfsingle['INJURIES_DIRECT'] + dfsingle['INJURIES_INDIRECT']).fillna(0)


# identify events with known location data
with_coordinates = df_details.BEGIN_LAT.notnull() & df_details.BEGIN_LON.notnull()
with_cz_name = df_details.CZ_NAME.notnull()
with_cz_fips = df_details.CZ_FIPS.notnull()
with_state_fips = df_details.STATE_FIPS.notnull()
with_state_name = df_details.STATE.notnull()
with_all_location_info = (
    df_details.CZ_NAME.notnull()
    & df_details.CZ_FIPS.notnull()
    & df_details.STATE_FIPS.notnull()
    & df_details.STATE.notnull()
    & df_details.BEGIN_LAT.notnull()
    & df_details.BEGIN_LON.notnull()
)

total_events = len(df_details)
events_with_coordinates = len(df_details[with_coordinates])

events_with_cz_name = len(df_details[with_cz_name])
events_with_cz_fips = len(df_details[with_cz_fips])
events_with_state_fips = len(df_details[with_state_fips])
events_with_state_name = len(df_details[with_state_name])
events_with_all_location_info = len(df_details[with_all_location_info])

perc_event_cz_name = events_with_cz_name / total_events * 100
perc_event_with_cz_fips = events_with_cz_fips / total_events * 100
perc_event_with_state_fips = events_with_state_fips / total_events * 100
perc_event_with_state_name = events_with_state_name / total_events * 100
perc_event_with_all_location_info = events_with_all_location_info / total_events * 100
perc_event_with_coordinates = events_with_coordinates / total_events * 100

print(f"total_events: {total_events}")
print(f"perc_event_cz_name: {perc_event_cz_name}")
print(f"perc_event_with_cz_fips: {perc_event_with_cz_fips}")
print(f"perc_event_with_state_fips: {perc_event_with_state_fips}")
print(f"perc_event_with_state_name: {perc_event_with_state_name}")
print(f"perc_event_with_all_location_info: {perc_event_with_all_location_info}")
print(f"perc_event_with_coordinates: {perc_event_with_coordinates}")

df_details = df_details.reindex(
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
        "ALL_INJURIES",
        "ALL_DEATHS",
        "ALL_DAMAGE",
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


# save full dataset, as csv and parquet
df_details = df_details.sort_values(
    ["BEGIN_DATETIME", "CZ_FIPS"], ascending=[True, True]
)
df_details = df_details.drop_duplicates()

df_details.to_csv(
    rf"{Output_Cleaned_Database_Path}\NCEI_Storm_Database_Cleaned_Details_1950-2024.csv",
    header=True,
    encoding="utf-8",
)
df_details.to_parquet(
    rf"{Output_Cleaned_Database_Path}\NCEI_Storm_Database_Cleaned_Details_1950-2024.parquet",
    compression="gzip",
)

# save version with just 1996 to 2024 data, as csv and parquet
df_details_1996_2024 = df_details[
    df_details["BEGIN_DATETIME"]
    >= datetime.strptime("1996-01-01 00:00:00", "%Y-%m-%d %H:%M:%S")
]
df_details_1996_2024 = df_details_1996_2024.sort_values(
    ["BEGIN_DATETIME", "CZ_FIPS"], ascending=[True, True]
)
df_details_1996_2024 = df_details_1996_2024.drop_duplicates()
# df_details_1996_2024.reset_index()

df_details_1996_2024.to_csv(
    rf"{Output_Cleaned_Database_Path}\NCEI_Storm_Database_Cleaned_Details_1996-2024.csv",
    header=True,
    encoding="utf-8",
)
df_details_1996_2024.to_parquet(
    rf"{Output_Cleaned_Database_Path}\NCEI_Storm_Database_Cleaned_Details_1996-2024.parquet",
    compression="gzip",
)
