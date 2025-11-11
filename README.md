# NCEI_Storm_Multihazard_Eventset
This repository contains scripts that allows the user to clean NOAA NCEI's Storm Events Database, and generate multi-hazard only and single hazard only eventset. Eventsets can be customized by defining time period (years), the multi-hazard timelag overlap (days), the hazard/peril event types (see database documentation), and various event impact threshold filters (deaths, injuries, crop damage, building damage).

The input database files necessary to run these scripts can be downloaded via HTML/FTP on the NCEI website at the below URL (as of Mar 2025).
- https://www.ncdc.noaa.gov/stormevents/ftp.jsp
- https://www.ncei.noaa.gov/pub/data/swdi/stormevents/csvfiles/
- ftp://ftp.ncei.noaa.gov/pub/data/swdi/stormevents/csvfiles/

To the best of my knowledge, this is the first publically avaliable Multihazard Eventset specific to the United States. Future work can adapt these scripts for the SHELDUS database led by Melanie Gall at ASU.

The formatting has been designed in a manner similar to the MYRIAD-HESA multihazard eventset developed by Judith Claassen.
https://github.com/judithclaassen/MYRIAD-HESA/
