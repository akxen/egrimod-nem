
# coding: utf-8

# # Extract Load and Dispatch Signals
# Data from AEMO's MMSDM database are used to construct historic load and dispatch signals. To illustrate how these signals can be constructed we extract data for one month as a sample. As the schema is the same for all MMSDM files, signals from different periods can be constructed by changing the csv file imported. Also, signals with longer time horizons can be constructed by chaining together data from multiple months.
# 
# ## Procedure
# 1. Import packages and declare paths to directories
# 2. Import datasets
# 3. Pivot dataframes extracting data from desired columns
# 4. Save data to file
# 
# ## Import packages

# In[1]:


import os
import pandas as pd


# ## Paths to directories

# In[2]:


# Core data directory (common files)
data_dir = os.path.abspath(os.path.join(os.path.curdir, os.path.pardir, os.path.pardir, 'data'))

# MMSDM data directory
mmsdm_dir = os.path.join(data_dir, 'AEMO', 'MMSDM')

# Output directory
output_dir = os.path.abspath(os.path.join(os.path.curdir, 'output'))


# ## Datasets
# ### MMSDM
# A summary of the tables used from AEMO's MMSDM database [1] is given below:
# 
# | Table | Description |
# | :----- | :----- |
# |DISPATCH_UNIT_SCADA | MW dispatch at 5 minute (dispatch) intervals for DUIDs within the NEM.|
# |TRADINGREGIONSUM | Contains load in each NEM region at 30 minute (trading) intervals.|
# 
# #### Unit Dispatch
# Parse and save unit dispatch data. Note that dispatch in MW is given at 5min intervals, and that the time resolution of demand data is 30min intervals, corresponding to the length of a trading period in the NEM. To align the time resolution of these signals unit dispatch data are aggregated, with mean power output over 30min intervals computed for each DUID.

# In[3]:


# Unit dispatch data
df_DISPATCH_UNIT_SCADA = pd.read_csv(os.path.join(mmsdm_dir, 'PUBLIC_DVD_DISPATCH_UNIT_SCADA_201706010000.CSV'),
                                     skiprows=1, skipfooter=1, engine='python')

# Convert to datetime objects
df_DISPATCH_UNIT_SCADA['SETTLEMENTDATE'] = pd.to_datetime(df_DISPATCH_UNIT_SCADA['SETTLEMENTDATE'])

# Pivot dataframe. Dates are the index values, columns are DUIDs, values are DUID dispatch levels
df_DISPATCH_UNIT_SCADA_piv = df_DISPATCH_UNIT_SCADA.pivot(index='SETTLEMENTDATE', columns='DUID', values='SCADAVALUE')

# To ensure the 30th minute interval is included during each trading interval the time index is offset
# by 1min. Once the groupby operation is performed this offset is removed.
df_DISPATCH_UNIT_SCADA_agg = df_DISPATCH_UNIT_SCADA_piv.groupby(pd.Grouper(freq='30Min', base=1, label='right')).mean()
df_DISPATCH_UNIT_SCADA_agg = df_DISPATCH_UNIT_SCADA_agg.set_index(df_DISPATCH_UNIT_SCADA_agg.index - pd.Timedelta(minutes=1))

df_DISPATCH_UNIT_SCADA_agg.to_csv(os.path.join(output_dir, 'signals_dispatch.csv'))


# #### Regional Load
# Load in each NEM region is given at 30min intervals.

# In[4]:


# Regional summary for each trading interval
df_TRADINGREGIONSUM = pd.read_csv(os.path.join(data_dir, 'AEMO', 'MMSDM', 'PUBLIC_DVD_TRADINGREGIONSUM_201706010000.CSV'),
                                  skiprows=1, skipfooter=1, engine='python')

# Convert settlement date to datetime
df_TRADINGREGIONSUM['SETTLEMENTDATE'] = pd.to_datetime(df_TRADINGREGIONSUM['SETTLEMENTDATE'])

# Pivot dataframe. Index is timestamp, columns are NEM region IDs, values are total demand
df_TRADINGREGIONSUM_piv = df_TRADINGREGIONSUM.pivot(index='SETTLEMENTDATE', columns='REGIONID', values='TOTALDEMAND')

df_TRADINGREGIONSUM_piv.to_csv(os.path.join(output_dir, 'signals_regional_load.csv'))


# ## References
# [1] - Australian Energy Markets Operator. Data Archive (2018). at http://www.nemweb.com.au/#mms-data-model
