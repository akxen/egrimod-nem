
# coding: utf-8

# # Compile Generator Data
# Technical and economic parameters for generators participating within Australia's National Electricity Market (NEM) are collated. AEMO's Market Management System Data Model (MMSDM) database tables [1] contain basic parameters such as registered capacities and generator types. The National Transmission Network Development Plan (NTNDP) database [2] contains data relating to unit technical parameters, operating costs, and heat rates. Information from these datasets are combined, culminating in one DataFrame that summarises relevant parameters for each unit within the NEM.
# 
# It should be noted that AEMO identifies individual generators within the NEM at differing levels of aggregation. When reporting unit dispatch AEMO identifies units by Dispatchable Unit Identifiers (DUIDs). In the following analysis DUIDs will be used as primary keys, allowing the generator dataset to be easily integrated with historic unit dispatch information.
# 
# ## Procedure
# 1. Import packages and declare paths to directories
# 2. Import datasets
# 3. Filter datasets
# 4. Combine information from datasets
# 5. Fill missing values
# 6. Save merged DataFrame to file
# 
# ## Import packages

# In[1]:


import os
import pickle

import numpy as np
import pandas as pd
import geopandas as gp
import kml2geojson


# ## Paths to directories

# In[2]:


# Core data directory (common files)
data_dir = os.path.abspath(os.path.join(os.path.curdir, os.path.pardir, os.path.pardir, 'data'))

# Network data directory
network_dir = os.path.abspath(os.path.join(os.path.curdir, os.path.pardir, '1_network'))

# MMSDM data directory
mmsdm_dir =os.path.join(data_dir, 'AEMO', 'MMSDM')

# Directory containing NTNDP files
ntndp_dir = os.path.join(data_dir, 'AEMO', 'NTNDP')

# Geoscience Australia power stations dataset
ga_path = os.path.join(data_dir, 'Geoscience_Australia', 'MajorPowerStations_v2', 'doc.kml')

# Output directory
output_dir = os.path.abspath(os.path.join(os.path.curdir, 'output'))


# ## Datasets
# ### MMSDM
# A summary of the tables used from AEMO's MMSDM database is given below:
# 
# | Table | Description |
# | :----- | :----- |
# |DUDETAILSUMMARY | Contains information regarding: schedule type (either scheduled, semi-scheduled, or non-scheduled), start-up type (either fast, slow, or not dispatched), and station IDs for each DUID.|
# |DUDETAIL | Contains information regarding registered capacities, maximum ramp-rates, and dispatch types (either generator or load) for each DUID. |
# |STATION | Contains information regarding station names, and addresses for each station ID.|
# |CO2EII_AVAILABLE_GENERATORS | Contains equivalent CO2 emissions intensities and fuel types for each genset. (Note: This table was obtained from AEMO's Current Reports database via the nemweb portal [3]|
# 
# ### NTNDP database
# A summary of the tables used from AEMO's 2016 NTNDP database is given below:
# 
# | Name | Filename | Description |
# | :---- | :-------- | :---- |
# |ACIL Allen - Technical parameters for units¶ | Fuel_and_Technology_Cost_Review_Data_ACIL_Allen.xlsx | ACIL Allen conducted a review of NEM generator parameters, with these data made available in AEMO's 2016 NTNDP database. Parameters such as startup costs, minimum up and down times, and minimum output as a percentage of nameplate capacity are given. |
# |NTNDP - Fuel Cost Profiles | 2016 Planning Studies - Additional Modelling Data and Assumptions summary.xlsm | Fuel cost profiles for different units / regions|
# |NTNDP - VOM | 2016 Planning Studies - Additional Modelling Data and Assumptions summary.xlsm | Variable Operating and Maintenance (VOM) costs for different units.|
# |NTNDP - Heat Rate | 2016 Planning Studies - Additional Modelling Data and Assumptions summary.xlsm | Heat rate data for different units.|
# 
# ### Generator cross-reference tables
# Different conventions have been adopted across datasets to identify units. In order to collate these data, cross-reference tables have been constructed. These tables link unit / generator IDs to DUIDs, allowing different datasets to be joined together.
# 
# | Name | Description |
# | :---- | :------|
# |DUID-NTNDP-GEN_PARAMS | Maps unit IDs used within ACIL_Allen's "Fuel_and_Technology_Cost_Review_Data_ACIL_Allen.csv" (found within AEMO's NTNDP database) to DUIDs.|
# |DUID-GA-POWER_STATIONS | Links DUIDs to generators in Geoscience Australia's Power Station dataset.|
# |DUID-NTNDP-FUEL_COST | Links DUIDs to fuel cost profiles found within "AEMO 2016 Planning Studies - Additional Modelling Data and Assumptions summary.xlsm" (from AEMO's 2016 NTNDP database).|
# |DUID-NTNDP-HEAT_RATES | Links DUIDs to heat rate data within "AEMO 2016 Planning Studies - Additional Modelling Data and Assumptions summary.xlsm" (from AEMO's 2016 NTNDP database).|
# |DUID-NTNDP-VOM | Links DUIDs to VOM data within "AEMO 2016 Planning Studies - Additional Modelling Data and Assumptions summary.xlsm" (from AEMO's 2016 NTNDP database).|

# In[3]:


# MMSDM data
# ----------
# DUID detail summary
df_DUDETAILSUMMARY = pd.read_csv(os.path.join(mmsdm_dir, 'PUBLIC_DVD_DUDETAILSUMMARY_201706010000.CSV'),
                         skiprows=1, skipfooter=1, engine='python')

# DUID details
df_DUDETAIL = pd.read_csv(os.path.join(mmsdm_dir, 'PUBLIC_DVD_DUDETAIL_201706010000.CSV'),
                         skiprows=1, skipfooter=1, engine='python')

# Station names
df_STATION = pd.read_csv(os.path.join(mmsdm_dir, 'PUBLIC_DVD_STATION_201706010000.CSV'),
                        skiprows=1, skipfooter=1, engine='python')

# Emissions factors for genset IDs
df_CO2EII = pd.read_csv(os.path.join(mmsdm_dir, 'CO2EII_AVAILABLE_GENERATORS_2017_43_20171103104827.CSV'),
                         skiprows=1, skipfooter=1, engine='python')

# NTNDP data
# ----------
# ACIL Allen data - unit commitment data
df_acil = pd.read_excel(os.path.join(ntndp_dir,'Fuel_and_Technology_Cost_Review_Data_ACIL_Allen.xlsx'),
                        sheet_name='Existing Generators', skiprows=1, index_col='Unit')

# Import NTNDP Fuel Cost Profiles
df_ntndp_coal = pd.read_excel(os.path.join(ntndp_dir, '2016 Planning Studies - Additional Modelling Data and Assumptions summary.xlsm'),
                              sheet_name='Coal Cost', skiprows=2, skip_footer=26, index_col='Fuel Cost ($/GJ)')

df_ntndp_gas = pd.read_excel(os.path.join(ntndp_dir, '2016 Planning Studies - Additional Modelling Data and Assumptions summary.xlsm'),
                             sheet_name='Gas Cost', skiprows=2, skip_footer=67, index_col='Fuel Cost ($/GJ)')

# Place all fuel cost data into one dataframe
df_ntndp_fuel_cost = df_ntndp_coal.append(df_ntndp_gas).add_prefix('Fuel_Cost_')

# Import VOM data from NTNDP database
df_ntndp_vom = pd.read_excel(os.path.join(ntndp_dir,'2016 Planning Studies - Additional Modelling Data and Assumptions summary.xlsm'),
                             sheet_name='VOM', skiprows=1, index_col='Generators')

# Import heat rate data
df_ntndp_hr = pd.read_excel(os.path.join(ntndp_dir,'2016 Planning Studies - Additional Modelling Data and Assumptions summary.xlsm'),
                            sheet_name='Heat Rates', skiprows=1, skip_footer=1326, index_col='Generators')

# Cross-reference tables
# ----------------------
# Mapping between DUIDs and ACIL Allen unit IDs
df_acil_cref = pd.read_csv(os.path.join(data_dir, 'cross_reference', 'DUID-NTNDP-GEN_PARAMS.csv'), index_col='DUID')

# Mapping between DUIDs and Geoscience Australia power stations
df_ga_cref = pd.read_csv(os.path.join(data_dir, 'cross_reference', 'DUID-GA-POWER_STATIONS.csv'), index_col='DUID')

# Mapping between DUIDs and NTNDP fuel cost profiles
df_fuel_cost_cref = pd.read_csv(os.path.join(data_dir, 'cross_reference', 'DUID-NTNDP-FUEL_COST.csv'), index_col='DUID')

# Mapping between DUIDs and NTNDP heat rates
df_heat_rate_cref = pd.read_csv(os.path.join(data_dir, 'cross_reference', 'DUID-NTNDP-HEAT_RATES.csv'), index_col='DUID')

# Mapping between DUIDs and NTNDP VOM costs
df_vom_cref = pd.read_csv(os.path.join(data_dir, 'cross_reference', 'DUID-NTNDP-VOM.csv'), index_col='DUID')


# ## Parse data
# For each dataset, change data types of columns as necessary, and filter records.

# In[4]:


def parse_CO2EII(df):
    return df.drop_duplicates(subset=['DUID']).set_index('DUID')
df_CO2EII_parsed = parse_CO2EII(df_CO2EII)

def parse_DUDETAILSUMMARY(df):
    """Extract DUID information from DUDETAILSUMMARY"""
    
    # Convert to datetime objects
    df['START_DATE'] = pd.to_datetime(df['START_DATE'])
    df['END_DATE'] = pd.to_datetime(df['START_DATE'])

    # Sort by end date and drop duplicates (retains most recent record)
    df = df.sort_values('END_DATE', ascending=False).drop_duplicates(subset=['DUID'])
    
    # Only keep records if end date is after the MMSDM volume date. This discards DUIDs that are no longer
    # within the dataset
    mask = df['END_DATE'] >= '2017-06-01 00:00:00'

    # Columns to keep
    cols = ['CONNECTIONPOINTID', 'REGIONID', 'STATIONID', 'PARTICIPANTID', 'TRANSMISSIONLOSSFACTOR',
            'DISTRIBUTIONLOSSFACTOR', 'SCHEDULE_TYPE', 'MIN_RAMP_RATE_UP', 'MIN_RAMP_RATE_DOWN',
            'MAX_RAMP_RATE_UP', 'MAX_RAMP_RATE_DOWN']

    return df.loc[mask].set_index('DUID').loc[:, cols]
df_DUDETAILSUMMARY_parsed = parse_DUDETAILSUMMARY(df_DUDETAILSUMMARY)

def parse_DUDETAIL(df):
    """Extract DUID information from DUDETAIL"""
    
    # Convert to datetime objects
    df['LASTCHANGED'] = pd.to_datetime(df['LASTCHANGED'])
    
    # Sort by LASTCHANGED and drop duplicates (keeping latest record for each DUID)
    df = df.sort_values('LASTCHANGED', ascending=False).drop_duplicates(subset=['DUID'])

    # Columns to keep
    cols = ['VOLTLEVEL', 'REGISTEREDCAPACITY', 'AGCCAPABILITY', 'DISPATCHTYPE', 'MAXCAPACITY', 'STARTTYPE',
            'NORMALLYONFLAG', 'PHYSICALDETAILSFLAG', 'SPINNINGRESERVEFLAG', 'INTERMITTENTFLAG', 'SEMISCHEDULE_FLAG', 
            'MAXRATEOFCHANGEUP', 'MAXRATEOFCHANGEDOWN']
    
    return df.set_index('DUID').loc[:, cols]
df_DUDETAIL_parsed = parse_DUDETAIL(df_DUDETAIL)

def parse_STATION(df):
    """Extract station names"""
    
    # Convert to datetime objects
    df['LASTCHANGED'] = pd.to_datetime(df['LASTCHANGED'])

    # Sort by LASTCHANGED and drop duplicates (keeping latest record for each DUID)
    df = df.sort_values('LASTCHANGED', ascending=False).drop_duplicates(subset=['STATIONID'])
    
    # Columns to keep
    cols = ['STATIONNAME', 'ADDRESS1', 'ADDRESS2', 'ADDRESS3', 'ADDRESS4', 'CITY', 'STATE', 'POSTCODE']
    
    return df.set_index('STATIONID').loc[:, cols]
df_STATION_parsed = parse_STATION(df_STATION)


# ## Join dataframes
# Procedure: 
# 1. Initialise dataframe, df_g, with DUDETAILSUMMARY data;
# 2. join DUDETAIL to df_g;
# 3. group CO2 emissions intensities within CO2EII by DUID. Compute the mean emissions intensity gensets constituting a DUID. Assign DUID emissions intensities to df_g;
# 4. assign fuel types to DUIDs within df_g;
# 5. merge station names with df_g;
# 6. join cross-reference table IDs to df_g;
# 6. merge ACIL Allen data with df_g;
# 7. merge fuel cost data with df_g;
# 8. merge VOM cost data with df_g;
# 9. merge heat rate data with df_g.

# In[5]:


# Join DUDETAILSUMMARY, DUDETAIL, and CO2EII
df_g = df_DUDETAILSUMMARY_parsed.join(df_DUDETAIL_parsed, how='left')
df_g['CO2E_EMISSIONS_FACTOR'] = df_CO2EII_parsed.groupby('DUID')['CO2E_EMISSIONS_FACTOR'].mean()
df_g = df_g.join(df_CO2EII_parsed[['CO2E_ENERGY_SOURCE']], how='left')

# Merge station names
df_g = pd.merge(df_g, df_STATION_parsed[['STATIONNAME']], left_on='STATIONID', right_index=True)

# Cross-reference keys
# --------------------
# Join ACIL Allen unit ID keys
df_g = df_g.join(df_acil_cref[['ACIL_ALLEN_UNIT_ID']], how='left')

# Join NTNDP fuel cost keys
df_g = df_g.join(df_fuel_cost_cref[['NTNDP_FUEL_COST_PROFILE']], how='left')

# Join NTNDP VOM cost keys
df_g = df_g.join(df_vom_cref[['NTNDP_VOM_ID']], how='left')

# Join NTNDP heat rate keys
df_g = df_g.join(df_heat_rate_cref[['NTNDP_HEAT_RATE_ID']], how='left')


# Join other datasets using cross-reference keys
# ----------------------------------------------
# ACIL Allen data
df_g = pd.merge(df_g, df_acil.add_prefix('ACIL_'), left_on='ACIL_ALLEN_UNIT_ID', right_index=True, how='left')

# Fuel cost data
df_g = pd.merge(df_g, df_ntndp_fuel_cost, left_on='NTNDP_FUEL_COST_PROFILE', right_index=True, how='left')

# NTNDP - VOM data
df_g = pd.merge(df_g, df_ntndp_vom.add_prefix('NTNDP_'), left_on='NTNDP_VOM_ID', right_index=True, how='left')

# NTNDP - Heat rate data
df_g = pd.merge(df_g, df_ntndp_hr[['Heat Rate (GJ/MWh)']].add_prefix('NTNDP_'), left_on='NTNDP_HEAT_RATE_ID', right_index=True, how='left')
df_g


# Note: Swanbank E Gas Turbine has no fuel cost data for 2016-2017. Using 2017-18 cost instead.

# In[6]:


# Using 2017-18 fuel cost data for Swanbank E power station
df_g.loc['SWAN_E','Fuel_Cost_2016-17'] = df_g.loc['SWAN_E', 'Fuel_Cost_2017-18']


# AEMO uses some DUIDs to represent dummy generators (DUIDs prefixed with "DG\_"), and reserve traders (DUIDs prefixed with "RT\_"). Also, some DUIDs refer to dispatchable loads. In this analysis only generators are considered, with loads, dummy generators, and reserve traders removed from the dataframe.

# In[7]:


mask = ~(df_g.index.str.startswith('DG_') | df_g.index.str.startswith('RT_')) & (df_g['DISPATCHTYPE'] == 'GENERATOR')
df_g = df_g[mask].copy()


# ### Fuel category
# Categorise generators by fuel type (either fossil, hydro, solar, wind, or biofuel).

# In[8]:


def assign_fuel_type(df):
    """Assign a fuel category to each generator"""
    
    # Fuel type field
    df['Fuel_Category'] = np.nan

    # Renewable types
    mask_renewable = df['CO2E_ENERGY_SOURCE'].isin(['Hydro', 'Solar', 'Wind'])
    df.loc[mask_renewable, 'Fuel_Category'] = df.loc[mask_renewable, 'CO2E_ENERGY_SOURCE']

    # Fossil fuel types
    mask_fossil = df['CO2E_ENERGY_SOURCE'].isin(['Diesel oil', 'Brown coal', 'Black coal', 'Kerosene - non aviation', 'Natural Gas (Pipeline)', 'Coal mine waste gas', 'Coal seam methane'])
    df.loc[mask_fossil, 'Fuel_Category'] = 'Fossil'

    # Biofuel types
    mask_biofuel = df['CO2E_ENERGY_SOURCE'].isin(['Landfill biogas methane', 'Bagasse', 'Biomass and industrial materials', 'Other Biofuels'])
    df.loc[mask_biofuel, 'Fuel_Category'] = 'Biofuel'
    
    return df
df_g = assign_fuel_type(df_g)


# ## Compute SRMC
# ### Thermal units
# 
# SRMC [\$/MWh] = VOM [\$/MWh] + (Fuel cost [\$/GJ]  $\times$  Heat Rate [GJ/MWh])
# 
# For the purposes of this analysis, 2016-17 fuel costs are used (except for Swanbank E Gas Turbine where no such value exists, and the value for 2017-18 has been used instead).
# 
# ### Solar, hydro, and wind
# No fuel cost for solar, hydro, and wind units. SRMC is only the VOM cost. Inspection of the NTNDP VOM cost data reveals that a majority of hydro units are assigned a cost of 7\$/MWh, and wind units 0\$/MWh. If no SRMC is assigned, then these values will be used by default.

# In[9]:


def assign_srmc(df):
    """Assign SRMCs to each generator"""
    
    # Fossil units
    mask_fossil = df_g['Fuel_Category'].isin(['Fossil'])
    df_g.loc[mask_fossil, 'SRMC'] = df_g.loc[mask_fossil, 'NTNDP_VOM ($/MWh)'] + (df_g.loc[mask_fossil, 'NTNDP_Heat Rate (GJ/MWh)'] * df_g.loc[mask_fossil, 'Fuel_Cost_2016-17'])

    # Renewable units
    mask_renewable = df_g['Fuel_Category'].isin(['Solar', 'Hydro', 'Wind'])
    df_g.loc[mask_renewable, 'SRMC'] = df_g.loc[mask_renewable, 'NTNDP_VOM ($/MWh)']

    # If a wind generator but has no SRMC assigned
    mask_wind = (df_g['Fuel_Category'] == 'Wind') & (pd.isnull(df_g['SRMC']))
    df_g.loc[mask_wind, 'SRMC'] = 0

    # If a hydro generator but has no SRMC assigned
    mask_hydro = (df_g['Fuel_Category'] == 'Hydro') & (pd.isnull(df_g['SRMC']))
    df_g.loc[mask_hydro, 'SRMC'] = 7

    return df_g
df_g = assign_srmc(df_g)


# Checking which DUIDs are still missing an SRMC.

# In[10]:


mask = df_g['SCHEDULE_TYPE'].isin(['SCHEDULED', 'SEMI-SCHEDULED']) & pd.isnull(df_g['SRMC'])
df_g.loc[mask]


# ## Assign DUIDs to nodes

# In[11]:


# Node data from network construction procedure
df_nodes = pd.read_csv(os.path.join(network_dir, 'output', 'network_nodes.csv'), index_col='NODE_ID')

# AEMO DUIDs - Geoscience Australia power station names
df_duid_stations = pd.read_csv(os.path.join(data_dir, 'cross_reference', 'DUID-GA-POWER_STATIONS.csv'), index_col='DUID')

# Power station - node assignments
df_station_nodes = pd.read_csv(os.path.join(network_dir, 'output', 'network_power_stations-nodes.csv'), index_col='POWER_STATION_ID', dtype={'NEAREST_NODE':np.int32})

# Join GA power station names with DUIDs
df_g = df_g.join(df_duid_stations[['GA_POWER_STATION_PLACEMARK_ID', 'GA_STATION_NAME']], how='left')
df_g.reset_index(inplace=True)

# Merge nodes assigned to GA power stations with DUIDs
df_g = pd.merge(df_g, df_station_nodes[['NEAREST_NODE', 'NEAREST_NODE_DISTANCE_KM', 'PLACEMARK_ID']], how='left', left_on='GA_POWER_STATION_PLACEMARK_ID', right_on='PLACEMARK_ID')


# Filtering and re-naming columns.

# In[12]:


# Columns to keep
cols = ['DUID', 'STATIONID', 'STATIONNAME', 'NEAREST_NODE', 'CO2E_ENERGY_SOURCE', 'Fuel_Category', 
        'CO2E_EMISSIONS_FACTOR', 'SCHEDULE_TYPE', 'REGISTEREDCAPACITY', 
        'ACIL_Min Gen (% of nameplate capacity)', 'ACIL_Ramp Up Rate (MW/h) - when running normally', 
        'ACIL_Ramp Down Rate (MW/h) - when running normally', 'ACIL_Ramp Up Rate (MW/h) - for start up', 
        'ACIL_Ramp Down Rate (MW/h) - for shut down', 'ACIL_Minimum On Time (Hours)', 'ACIL_Minimum Off Time (Hours)', 
        'ACIL_Cold Start-up Costs ($/MW)', 'ACIL_Warm Start-up Costs ($/MW)', 'ACIL_Hot Start-up Costs ($/MW)', 
        'NTNDP_VOM ($/MWh)', 'NTNDP_Heat Rate (GJ/MWh)', 'ACIL_No Load Fuel Consumption (% of Full Load Fuel Consumption)', 
        'Fuel_Cost_2016-17', 'SRMC']

# Re-naming selected columns
rename_cols = {'NEAREST_NODE':'NODE', 'CO2E_ENERGY_SOURCE':'FUEL_TYPE', 'Fuel_Category':'FUEL_CAT', 
        'CO2E_EMISSIONS_FACTOR':'EMISSIONS', 'REGISTEREDCAPACITY':'REG_CAP',
        'ACIL_Min Gen (% of nameplate capacity)':'MIN_GEN', 'ACIL_Ramp Up Rate (MW/h) - when running normally':'RR_UP', 
        'ACIL_Ramp Down Rate (MW/h) - when running normally':'RR_DOWN', 'ACIL_Ramp Up Rate (MW/h) - for start up':'RR_STARTUP', 
        'ACIL_Ramp Down Rate (MW/h) - for shut down':'RR_SHUTDOWN', 'ACIL_Minimum On Time (Hours)':'MIN_ON_TIME', 
        'ACIL_Minimum Off Time (Hours)':'MIN_OFF_TIME', 'ACIL_Cold Start-up Costs ($/MW)':'SU_COST_COLD', 
        'ACIL_Warm Start-up Costs ($/MW)':'SU_COST_WARM', 'ACIL_Hot Start-up Costs ($/MW)':'SU_COST_HOT', 
        'NTNDP_VOM ($/MWh)':'VOM', 'NTNDP_Heat Rate (GJ/MWh)':'HEAT_RATE', 'Fuel_Cost_2016-17':'FC_2016-17',
        'ACIL_No Load Fuel Consumption (% of Full Load Fuel Consumption)':'NL_FUEL_CONS', 'SRMC':'SRMC_2016-17'}

# Merge information describing nodes to which DUIDs are connected
df_g_sav = df_g[cols].merge(df_nodes[['NEM_REGION', 'NEM_ZONE']], left_on='NEAREST_NODE', right_index=True, how='left').rename(columns=rename_cols)

# Express min gen in terms of MW
df_g_sav['MIN_GEN'] = (df_g_sav['MIN_GEN'] / 100) * df_g['REGISTEREDCAPACITY']

# Express startup costs in terms of $
def get_startup_cost(row):
    return pd.Series({'SU_COST_COLD':row['SU_COST_COLD'] * row['REG_CAP'],
                      'SU_COST_WARM':row['SU_COST_WARM'] * row['REG_CAP'],
                      'SU_COST_HOT':row['SU_COST_HOT'] * row['REG_CAP']})
df_g_sav[['SU_COST_COLD', 'SU_COST_WARM', 'SU_COST_HOT']] = df_g_sav.apply(get_startup_cost, axis=1)

# Express no-load fuel consumption % as a fraction and fill missing entries for hydro plant
df_g_sav['NL_FUEL_CONS'] = df_g_sav['NL_FUEL_CONS'] / 100
df_g_sav['NL_FUEL_CONS'] = df_g_sav.apply(lambda x: 0 if x['FUEL_CAT'] in ['Hydro', 'Solar'] else x['NL_FUEL_CONS'], axis=1)

# Re-order columns
cols = ['DUID', 'STATIONID', 'STATIONNAME', 'NEM_REGION', 'NEM_ZONE', 'NODE', 'FUEL_TYPE', 'FUEL_CAT', 'EMISSIONS',
       'SCHEDULE_TYPE', 'REG_CAP', 'MIN_GEN', 'RR_STARTUP', 'RR_SHUTDOWN', 'RR_UP', 'RR_DOWN', 'MIN_ON_TIME',
       'MIN_OFF_TIME', 'SU_COST_COLD', 'SU_COST_WARM', 'SU_COST_HOT', 'VOM', 'HEAT_RATE', 'NL_FUEL_CONS', 'FC_2016-17', 'SRMC_2016-17']
df_g_sav = df_g_sav[cols]


# ### Clean-up missing entries

# Some heat rates are not provided for hydro units. For completeness, these values are set to 0 in the final dataset.

# In[13]:


# Clean-up missing heat rate values
df_g_sav['HEAT_RATE'] = df_g_sav.apply(lambda x: 0 if x['FUEL_CAT'] in ['Hydro'] else x['HEAT_RATE'], axis=1)


# Renewables have no fuel costs. Fuel cost entries are set to zero for these generators.

# In[14]:


# Clean-up missing fuel costs for solar, wind, and hydro units
df_g_sav['FC_2016-17'] = df_g_sav.apply(lambda x: 0 if x['FUEL_CAT'] in ['Wind', 'Hydro', 'Solar', 'Biofuel'] else x['FC_2016-17'], axis=1)


# The NTNDP database provides no information regarding solar plant. MMSDM tables do provide information regarding maximum ramp-up and ramp-down rates for solar plant (in MW/min), and it is these values which have been used instead. Minimum on and off times are set to 0 hours, and these plant are assumed to have start-up costs of $0.

# In[15]:


# Clean-up missing info for solar generators
def clean_solar_records(row):
    if row['FUEL_CAT'] in ['Solar']:
        row['MIN_GEN'] = 0
        row['RR_STARTUP'] = df_DUDETAILSUMMARY_parsed.loc[row['DUID'], 'MAX_RAMP_RATE_UP'] * 60
        row['RR_SHUTDOWN'] = df_DUDETAILSUMMARY_parsed.loc[row['DUID'], 'MAX_RAMP_RATE_DOWN'] * 60
        row['RR_UP'] = df_DUDETAILSUMMARY_parsed.loc[row['DUID'], 'MAX_RAMP_RATE_UP'] * 60
        row['RR_DOWN'] = df_DUDETAILSUMMARY_parsed.loc[row['DUID'], 'MAX_RAMP_RATE_DOWN'] * 60
        row['MIN_ON_TIME'] = 0
        row['MIN_OFF_TIME'] = 0
        row['SU_COST_COLD'] = 0
        row['SU_COST_WARM'] = 0
        row['SU_COST_HOT'] = 0
    return row
df_g_sav = df_g_sav.apply(clean_solar_records, axis=1)


# Inspection of the dataset reveals that two DUIDs, YARWUN_1 and SITHE01 do not have minimum on and off time parameters within the NTNDP database. As both are gas plant, it is assumed they can be started / shutdown quickly, and the minimum on / off times are also short. Here it is assumed that these plant have minimum on / off times of 1hr, similar to other gas plant in the dataset.

# In[16]:


# Clean up missing entries for YARWUN_1 and SITHE01
def clean_yarwun_and_sithe_records(row):
    if row['DUID'] in ['YARWUN_1', 'SITHE01']:
        row['MIN_ON_TIME'] = 1
        row['MIN_OFF_TIME'] = 1
    return row
df_g_sav = df_g_sav.apply(clean_yarwun_and_sithe_records, axis=1)


# ## Save data

# In[17]:


# Only keep scheduled and semi-scheduled generators that have been assigned to a node.
df_g_sav.set_index('DUID', inplace=True)
mask = df_g_sav['SCHEDULE_TYPE'].isin(['SCHEDULED', 'SEMI-SCHEDULED']) & ~pd.isnull(df_g_sav['NODE'])

# Write to csv file
df_g_sav[mask].to_csv(os.path.join(output_dir, 'generators.csv'))


# ### Identify DUIDs missing from the final dataset

# In[18]:


# DUIDs in original dataset
mask = df_g['SCHEDULE_TYPE'].isin(['SCHEDULED', 'SEMI-SCHEDULED'])
original_duids = set(df_g[mask].set_index('DUID').index)

# DUIDs in final dataset
mask = df_g_sav['SCHEDULE_TYPE'].isin(['SCHEDULED', 'SEMI-SCHEDULED']) & ~pd.isnull(df_g_sav['NODE'])
final_duids =  set(df_g_sav[mask].index)

original_duids - final_duids


# ### Compare to AEMO Generation Information

# In[19]:


def extract_aemo_generation_info(fname, nem_region):
    """Extract AEMO generator information report data"""
    
    df = pd.read_excel(os.path.join(data_dir, 'AEMO', 'generation_information', fname),
                      sheet_name='Existing S & SS Generation', skiprows=1, skipfooter=1)
    df['NEM_REGION'] = nem_region
    return df

# Concatenate generator information from each of the five NEM regions
df = pd.DataFrame()
for region in ['NSW', 'QLD', 'SA', 'TAS', 'VIC']:
    fname = ''.join(['Generation_Information_', region ,'_05062017.xlsx'])
    df = pd.concat([df, extract_aemo_generation_info(fname, ''.join([region, '1']))])

# Mapping fuel types to generic generation technologies
gen_types = {'Kerosene - non aviation':'OCGT/CCGT', 'Natural Gas (Pipeline)':'OCGT/CCGT', 'Natural Gas Pipeline':'OCGT/CCGT',
             'Coal seam methane':'OCGT/CCGT', 'Fuel Oil':'OCGT/CCGT', 'Coal Seam Methane':'OCGT/CCGT', 'Diesel':'OCGT/CCGT',
             'Kerosene Aviation fuel used for stationary energy - avtur':'OCGT/CCGT', 'Brown Coal':'Coal',
             'Diesel oil':'OCGT/CCGT', 'Brown coal':'Coal', 'Black coal':'Coal', 'Black Coal': 'Coal', 'Water':'Hydro',
             'Hydro':'Hydro', 'Solar':'Solar', 'Wind':'Wind'}
df['GEN_TYPE'] = df['Fuel Type'].map(lambda x: gen_types[x])

# Aggregating by region and generation type 
# -----------------------------------------
# AEMO generation information report data
df1 = df.groupby(['NEM_REGION', 'GEN_TYPE'])['Installed Capacity (MW)'].sum()

# Compiled dataset
mask = df_g_sav['SCHEDULE_TYPE'].isin(['SCHEDULED', 'SEMI-SCHEDULED']) & ~pd.isnull(df_g_sav['NODE'])
df2 = df_g_sav[mask].copy()
df2['GEN_TYPE'] = df2['FUEL_TYPE'].map(lambda x: gen_types[x])
df3 = df2.groupby(['NEM_REGION', 'GEN_TYPE'])['REG_CAP'].sum()

df1.reset_index()
df3.reset_index()

df4 = pd.merge(df3.reset_index(), df1.reset_index(), left_on=['NEM_REGION', 'GEN_TYPE'], right_on=['NEM_REGION', 'GEN_TYPE']).rename(columns={'Installed Capacity (MW)':'AEMO Gen. Info.', 'REG_CAP':'Gen. Dataset'})
df4['Delta'] = df4['Gen. Dataset'] - df4['AEMO Gen. Info.']
df4['Percent Difference'] = 100 * (df4['Delta']  / df4['Gen. Dataset'])
df4.to_latex(os.path.join(output_dir, 'tables', 'generating_capacity_comparison_table.tex'))


# ## (Data used to construct cross-reference tables)
# The block below creates a template that is used to construct cross-reference tables. The template is based on DUDETAILSUMMARY, and also adds information regarding station names, fuel types, and registered capacities. These fields assist in the data validation process, allowing multiple fields to be checked when linking generator IDs in various datasets to DUIDs.
# 
# The code also extracts generator information from Geoscience Australia's power station database, and writes these data to a csv file. This file is used to link power stations in Geoscience Australia's datasets to DUIDs.

# In[20]:


def create_cross_reference_template():
    """Create template used to link generator IDs in other datasets to DUIDs"""
    
    # Use DUDETAILSUMMARY as a base. Add STATIONID, SCHEDULE_TYPE, REGIONID
    df = df_DUDETAILSUMMARY_parsed[['STATIONID', 'SCHEDULE_TYPE', 'REGIONID']]
    
    # Join STATIONNAME using STATIONID
    df = df.merge(df_STATION_parsed[['STATIONNAME']], how='left', left_on='STATIONID', right_index=True)
    
    # Join CO2E_ENERGY_SOURCE (fuel type) using DUID
    df = df.join(df_CO2EII_parsed[['CO2E_ENERGY_SOURCE']], how='left')
    
    # Join REGISTEREDCAPACITY using DUID
    df = df.join(df_DUDETAIL_parsed[['DISPATCHTYPE', 'REGISTEREDCAPACITY']], how='left')

    # Re-order columns
    df = df[['STATIONID', 'STATIONNAME', 'REGIONID', 'REGISTEREDCAPACITY', 'CO2E_ENERGY_SOURCE', 'SCHEDULE_TYPE']]
    
    return df
df_template = create_cross_reference_template()
df_template.to_csv(os.path.join(output_dir, 'cross_reference_template', 'cross_reference_template.csv'))

# Convert Geoscience Australia power stations kml file to geojson format
kml_path = os.path.join(data_dir, 'Geoscience_Australia', 'MajorPowerStations_v2', 'doc.kml')
kml2geojson.main.convert(kml_path, os.path.join(output_dir, 'cross_reference_template', 'geojson'))

# Load power station geojson dataset using geopandas
gdf_p = gp.read_file(os.path.join(output_dir, 'cross_reference_template', 'geojson', 'doc.geojson'))

# Save Geoscience Australia generator data to spreadsheet
gdf_p[['id', 'name', 'STATE', 'PRIMARYFUELTYPE', 'GENERATIONMW']].set_index('id').to_csv(os.path.join(output_dir, 'cross_reference_template', 'Geoscience_Australia_power_stations.csv'))


# ## References

# [1] - Australian Energy Markets Operator. Data Archive (2018). at http://www.nemweb.com.au/#mms-data-model:download
# 
# [2] - Australian Energy Markets Operator. NTNDP Database. (2018). at https://www.aemo.com.au/Electricity/National-Electricity-Market-NEM/Planning-and-forecasting/National-Transmission-Network-Development-Plan/NTNDP-database
# 
# [3] - Australian Energy Markets Operator. Current Reports. (2018). at http://www.nemweb.com.au/Reports/Current/
