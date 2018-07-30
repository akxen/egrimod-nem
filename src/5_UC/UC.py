
# coding: utf-8

# # Unit Commitment Model of Australia's National Electricity Market
# A unit commitmnet (UC) model of Australia's National Electricity Market (NEM) is developed using the generator and network datasets previously constructed. The UC model is based on [1], and takes into account a number of technical and economic parameters associated with power system operation. Compared to models with linear cost minimisation objective functions, UC models introduce a number of additional continuous and binary variables, resulting in a mixed integer linear program (MILP). The complexity of solving UC models is also increased as, unlike DCOPF models with linear objective functions which are solved sequentially for each time period, UC models solve for an optimal dispatch schedule for each unit over the time interval under investigation. Computer memory limiations were encountered when attempting to use the full network represenatation with the UC formulation adopted here. For this reason a simplified representation of the NEM's network is constructed. This representation makes use of NEM zones which are used in AEMO's own market models [2]. Rather than considering power balance constraints for each node, nodes are aggregated into their respective NEM zones. Power balance constraints are then enforced for each zone. This network representation is similar to the network representation of the NEM provided in [3], limiting power flows between zones according to interconnector capabilities.
# 
# The model presented here uses MMSDM data for June 2017 as an example. Data for only one month is loaded into this Jupyter Notebook to reduce the compuational burden of storing many large MMSDM tables in memory. As the schema of MMSDM tables is time invariant, other periods can easily be analysed by loading alternative MMSDM tables and selecting different intervals. When solving the UC model, a 24hr period (48 trading intervals) is investigated. The results from the optimisation problem are pickled and saved.
# 
# A summary of the steps taken to construct the UC model is as follows:
# 1. Import pacakges, declare paths to files, and load data
# 2. Organise model data:
#     * summarise important parameters for each node (e.g. assigned DUIDs, NEM region, NEM zone, proportion of regional demand consumed at node);
#     * compute aggregate nodal power injections from intermittent sources at each node for each time interval.
# 3. Aggregate data for nodes by NEM zones
# 4. Construct branch incidence matrix for AC lines
# 7. Construct incidence matrix describing the connections of HVDC links
# 8. Construct UC model
# 9. Solve UC model for time interval under investigation
# 

# ## Import packages

# In[1]:


import os
import pickle
from math import pi
import random

import numpy as np
import pandas as pd
import geopandas as gp

from pyomo.environ import *
import matplotlib.pyplot as plt


# ## Declare paths to files

# In[2]:


# Core data directory
data_dir = os.path.abspath(os.path.join(os.path.curdir, os.path.pardir, os.path.pardir, 'data'))

# Network directory
network_dir = os.path.abspath(os.path.join(os.path.curdir, os.path.pardir, '1_network'))

# Generators directory
gens_dir = os.path.abspath(os.path.join(os.path.curdir, os.path.pardir, '2_generators'))

# Signals directory
signals_dir = os.path.abspath(os.path.join(os.path.curdir, os.path.pardir, '3_load_and_dispatch_signals'))

# Output path
output_dir = os.path.abspath(os.path.join(os.path.curdir, 'output'))


# ##  Load data

# In[3]:


# Network data
# ------------
# Nodes
df_n = pd.read_csv(os.path.join(network_dir, 'output', 'network_nodes.csv'), index_col='NODE_ID', dtype={'NEAREST_NODE':np.int32})

# Edges
df_e = pd.read_csv(os.path.join(network_dir, 'output', 'network_edges.csv'), index_col='LINE_ID')

# HVDC links
df_hvdc = pd.read_csv(os.path.join(network_dir, 'output', 'network_hvdc_links.csv'), index_col='HVDC_LINK_ID')

# AC interconnector links
df_ac_i = pd.read_csv(os.path.join(network_dir, 'output', 'network_ac_interconnector_links.csv'), index_col='INTERCONNECTOR_ID')

# AC interconnector limits
df_ac_ilim = pd.read_csv(os.path.join(network_dir, 'output', 'network_ac_interconnector_flow_limits.csv'), index_col='INTERCONNECTOR_ID')

# Power station - node assignments
df_station_nodes = pd.read_csv(os.path.join(network_dir, 'output', 'network_power_stations-nodes.csv'), index_col='POWER_STATION_ID', dtype={'NEAREST_NODE':np.int32})


# Generator data
# --------------
df_g = pd.read_csv(os.path.join(gens_dir, 'output', 'generators.csv'), index_col='DUID')


# Dispatch and load signals
# -------------------------
# Dispatch signals from SCADA data
df_scada = pd.read_csv(os.path.join(signals_dir, 'output', 'signals_dispatch.csv'), index_col='SETTLEMENTDATE', parse_dates=['SETTLEMENTDATE'])

# Regional demand signals
df_regd = pd.read_csv(os.path.join(signals_dir, 'output', 'signals_regional_load.csv'), index_col='SETTLEMENTDATE', parse_dates=['SETTLEMENTDATE'])


# Cross-reference tables
# ----------------------
# AEMO DUIDs - Geoscience Australia power station names
df_duid_stations = pd.read_csv(os.path.join(data_dir, 'cross_reference', 'DUID-GA-POWER_STATIONS.csv'), index_col='DUID')


# ## Organise model data
# ### Summarise model data for each node
# The steps taken to collate data used in the UC model are as follows:
# 1. Initialise a dataframe, `df_m`, that will be used to summarise UC model data at each node
# 2. Assign DUIDs to each node
# 3. Create dataframe that contains intermittent power injections (from wind and solar) at each node for each time period.
# 4. Create dataframe that contains demand for each NEM region for each time period
# 5. Construct branch incidence matrix for the network

# In[4]:


# Dataframe that will contain a summary of data used in the DCOPF model
df_m = df_n.copy()

# DUIDs assigned to each node
df_m['DUID'] = df_g.reset_index().groupby('NODE')[['DUID']].aggregate(lambda x: set(x)).reindex(df_m.index, fill_value=set())
df_m


# ### Intermittent generation at each node
# DUIDs corresponding to intermittent generators (wind and solar) are identified, and the net power injection at each node from these generators computed. Nodal power injections from intermittent sources are then aggregated by NEM zone.
# 
# Note: Unit dispatch for some DUIDs is negative. This could be a result of the measurement methods used to collect dispatch data. As these negative values are small, they are unlikely to have a material impact on the final output of the model. However, the declaration of parameters in the UC model to follow may raise warnings when encountering negative values. To prevent this from occurring, these negative values are set to zero.

# In[5]:


# Find all intermittent DUIDs (wind and solar)
mask = df_g['FUEL_CAT'].isin(['Wind', 'Solar']) 
ig_ids = df_g[mask].index

# Find total intermittent generation at each node for each time period
df_inter = df_scada.reindex(ig_ids, axis=1, fill_value=0).T.join(df_g[['NEM_ZONE']], how='left').groupby('NEM_ZONE').sum().T

# Set negative dispatch values to 0
mask = df_inter < 0
df_inter[mask] = 0

# Re-index, so all nodes are contained within columns
df_inter = df_inter.reindex(df_n['NEM_ZONE'].unique(), axis=1, fill_value=0)
df_inter.head()


# ### Total demand in each NEM region
# Total demand for each NEM region.

# In[6]:


df_regd.head()


# ## Summarise model data for reduced network

# In[7]:


# Initialise dataframe that will contain model data for the reduced network
df_rm = pd.DataFrame(index=df_m['NEM_ZONE'].unique())

# Proportion of NEM region demand for each zone
df_rm = df_m.groupby('NEM_ZONE')[['PROP_REG_D']].sum()

# NEM region associated with each NEM zone
df_rm['NEM_REGION'] = df_rm.apply(lambda x: df_m.drop_duplicates('NEM_ZONE').set_index('NEM_ZONE').loc[x.name , 'NEM_REGION'], axis=1)
df_rm['DUID'] = df_g.reset_index().groupby('NEM_ZONE')['DUID'].aggregate(lambda x: set(x))
df_rm.head()


# ### Demand for each NEM zone

# In[8]:


# Initialise matrix containing zonal demand
df_zd = pd.DataFrame(index=df_rm.index, columns=df_regd.index)

def get_zone_demand(row):
    # NEM region corresponding to NEM zone
    nem_region = df_m.drop_duplicates('NEM_ZONE').set_index('NEM_ZONE').loc[row.name, 'NEM_REGION']
    
    # Demand series for NEM zone's corresponding NEM region
    region_demand = df_regd.loc[:, nem_region]
    
    # Demand series for each NEM zone
    zone_demand = region_demand * df_rm.loc[row.name, 'PROP_REG_D']
    return zone_demand

df_zd = df_zd.apply(get_zone_demand, axis=1).T
df_zd.head()


# ### Incidence matrix for AC connections between NEM zones

# In[9]:


# Lines in the reduced network
rn_lines = ['ADE,NSA', 'SESA,ADE', 'MEL,SESA', 'MEL,CVIC', 'MEL,LV', 'MEL,NVIC', 'CVIC,SWNSW', 'NVIC,SWNSW', 
            'NVIC,CAN', 'CAN,SWNSW', 'CAN,NCEN', 'NCEN,NNS', 'NNS,SWQ', 'SWQ,SEQ', 'SWQ,CQ', 'SEQ,CQ', 'CQ,NQ', 'NNS,SEQ']

# Incidence matrix for network based on NEM zones as nodes
df_ac_C = pd.DataFrame(index=rn_lines, columns=df_rm.index, data=0)

for line in rn_lines:
    # Get 'from' and 'to' nodes for each line
    fn, tn = line.split(',')
    
    # Assign 'from' node value of 1
    df_ac_C.loc[line, fn] = 1
    
    # Assign 'to' node value of -1
    df_ac_C.loc[line, tn] = -1
df_ac_C.head()


# ### Incidence matrix for HVDC links

# In[10]:


# Drop Directlink HVDC line (techincally not an interconnector)
df_hvdc = df_hvdc.drop('DIRECTLINK')

# Assign 'from' and 'to' zones to each HVDC link
df_hvdc['FROM_ZONE'] = df_hvdc.apply(lambda x: df_n.loc[x['FROM_NODE'], 'NEM_ZONE'], axis=1)
df_hvdc['TO_ZONE'] = df_hvdc.apply(lambda x: df_n.loc[x['TO_NODE'], 'NEM_ZONE'], axis=1)

# Incidence matrix for HVDC links
df_hvdc_C = pd.DataFrame(index=df_hvdc.index, columns=df_rm.index, data=0)

for index, row in df_hvdc.iterrows():
    # Extract 'from' and 'to' zones for each HVDC link
    fz, tz = row['FROM_ZONE'], row['TO_ZONE']
    
    # Assign value of 1 to 'from' zones
    df_hvdc_C.loc[index, fz] = 1
    
    # Assign value of -1 to 'to' zones
    df_hvdc_C.loc[index, tz] = -1
df_hvdc_C


# ## Minimum Reserve levels for each NEM region
# Minimum reserve levels in MW for each NEM region are obtained from [4].

# In[11]:


df_mrl = pd.Series(data={'NSW1': 673.2, 'QLD1': 666.08, 'SA1': 195, 'TAS1': 194, 'VIC1': 498})


# ## Unit Commitment Model

# In[12]:


def run_uc_model(fname, fix_hydro=True):
    # Model
    # -----
    m = ConcreteModel()

    # Sets
    # ----
    # Generators
    if fix_hydro:
        mask = (df_g['SCHEDULE_TYPE'] == 'SCHEDULED')  & df_g['FUEL_CAT'].isin(['Fossil']) & ~pd.isnull(df_g['MIN_ON_TIME'])
        m.G = Set(initialize=df_g[mask].index)

        mask = (df_g['SCHEDULE_TYPE'] == 'SCHEDULED')  & df_g['FUEL_CAT'].isin(['Hydro']) & ~pd.isnull(df_g['MIN_ON_TIME'])
        m.HYDRO = Set(initialize=df_g[mask].index)

    else:
        mask = (df_g['SCHEDULE_TYPE'] == 'SCHEDULED')  & df_g['FUEL_CAT'].isin(['Hydro', 'Fossil']) & ~pd.isnull(df_g['MIN_ON_TIME'])
        m.G = Set(initialize=df_g[mask].index)
        m.HYDRO = Set()

    # NEM zones
    m.J = Set(initialize=df_rm.index)
    
    # NEM regions
    m.S = Set(initialize=df_rm['NEM_REGION'].unique())
    
    # HVDC links
    m.H = Set(initialize=df_hvdc_C.index)
    
    # AC lines
    m.A = Set(initialize=df_ac_C.index)

    # (Dictionary to convert between timestamps and time indices)
    t_dict = {j+1: k for j, k in enumerate(df_zd[:48].index)}
    
    # Time indices
    m.T = Set(initialize=list(t_dict.keys()), ordered=True)

    # Parameters
    # ----------
    # Constant linear variable cost (SRMC)
    def C_LV_rule(m, g):
        return float(df_g.loc[g, 'SRMC_2016-17'])
    m.C_LV = Param(m.G, initialize=C_LV_rule)

    # No load cost
    def C_NL_rule(m, g):
        if df_g.loc[g, 'FUEL_CAT'] == 'Hydro':
            return float(0)
        else:
            return float((df_g.loc[g, 'NL_FUEL_CONS'] * df_g.loc[g, 'HEAT_RATE'] * df_g.loc[g, 'FC_2016-17']))
    m.C_NL = Param(m.G, initialize=C_NL_rule)

    # Maximum power output
    def P_MAX_rule(m, g):
        return float(df_g.loc[g, 'REG_CAP'])
    m.P_MAX = Param(m.G, initialize=P_MAX_rule)

    # Minimum power output
    def P_MIN_rule(m, g):
        min_gen = df_g.loc[g, 'MIN_GEN']

        # If no data for min gen as % of nameplate capacity, return 0
        if pd.isnull(min_gen):
            return float(0)
        else:
            return float(min_gen)
    m.P_MIN = Param(m.G, initialize=P_MIN_rule)

    # Hydro output
    def HYDRO_OUTPUT_rule(m, hydro, t):
        return float(df_scada.loc[t_dict[t], hydro])
    m.HYDRO_OUTPUT = Param(m.HYDRO, m.T, initialize=HYDRO_OUTPUT_rule)

    # Time interval length in hours
    m.DELT = Param(initialize=0.5)

    # Ramp up capability
    def RU_rule(m, g):
        return float(df_g.loc[g, 'RR_UP'] * m.DELT.value)
    m.RU = Param(m.G, initialize=RU_rule)

    # Ramp down capability
    def RD_rule(m, g):
        return float(df_g.loc[g, 'RR_DOWN'] * m.DELT.value)
    m.RD = Param(m.G, initialize=RD_rule)

    # Minimum off time expressed in terms of time intervals
    def TD_rule(m, g):
        return float(df_g.loc[g, 'MIN_OFF_TIME'] / m.DELT.value)
    m.TD = Param(m.G, initialize=TD_rule)

    # Minimum on time expressed in terms of time intervals
    def TU_rule(m, g):
        return float(df_g.loc[g, 'MIN_ON_TIME'] / m.DELT.value)
    m.TU = Param(m.G, initialize=TU_rule)

    # Demand for each NEM zone
    def D_rule(m, j, t):
        return float(df_zd.loc[t_dict[t], j])
    m.D = Param(m.J, m.T, initialize=D_rule)

    # HVDC incidence matrix
    def HVDC_C_rule(m, h, j):
        return float(df_hvdc_C.loc[h, j])
    m.HVDC_C = Param(m.H, m.J, initialize=HVDC_C_rule)
    
    # AC incidence matrix
    def AC_C_rule(m, a, j):
        return float(df_ac_C.loc[a, j])
    m.AC_C = Param(m.A, m.J, initialize=AC_C_rule)

    # Intermittent generation for each zone
    def P_W_rule(m, j, t):
        wind_output = float(df_inter.loc[t_dict[t], j])
        if wind_output < 0:
            return float(0)
        else:
            return wind_output
    m.P_W = Param(m.J, m.T, initialize=P_W_rule)

    # Start-up costs
    def C_SU_rule(m, g):
        return float(df_g.loc[g, 'SU_COST_COLD'] * m.P_MIN[g])
    m.C_SU = Param(m.G, initialize=C_SU_rule)

    # Start-up ramp rate
    def SU_D_rule(m, g):
        ru_intervals = m.P_MIN[g] / (df_g.loc[g, 'RR_STARTUP'] * m.DELT.value)
        if ru_intervals > 1:
            return float(ceil(ru_intervals))
        else:
            return float(0) 
    m.SU_D = Param(m.G, initialize=SU_D_rule)

    # Shutdown ramp rate
    def SD_D_rule(m, g):
        rd_intervals = m.P_MIN[g] / (df_g.loc[g, 'RR_SHUTDOWN'] * m.DELT.value)
        if rd_intervals > 1:
            return float(ceil(rd_intervals))
        else:
            return float(0)
    m.SD_D = Param(m.G, initialize=SD_D_rule)

    # Startup capability
    def SU_rule(m, g):
        if m.SU_D[g]:
            return m.P_MIN[g]
        else:
            return float(df_g.loc[g, 'RR_STARTUP'])
    m.SU = Param(m.G, initialize=SU_rule)

    # Shutdown capability
    def SD_rule(m, g):
        if m.SD_D[g]:
            return m.P_MIN[g]
        else:
            return float(df_g.loc[g, 'RR_SHUTDOWN'])
    m.SD = Param(m.G, initialize=SD_rule)

    # Minimum reserve level (up) for each NEM zone
    def D_UP_rule(m, s):
        return float(df_mrl.loc[s])
    m.D_UP = Param(m.S, initialize=D_UP_rule)

    # Minimum reserve level (down) for each NEM zone
    def D_DOWN_rule(m, s):
        return float(df_mrl.loc[s] / 10)
    m.D_DOWN = Param(m.S, initialize=D_DOWN_rule)


    # Variables
    # ---------
    # One state for generator
    m.u = Var(m.G, m.T, within=Binary)
    
    # Startup indicator
    m.v = Var(m.G, m.T, within=Binary)
    
    # Shutdown indicator
    m.w = Var(m.G, m.T, within=Binary)
    
    # Reserve (up) for each generator
    m.r_up = Var(m.G, m.T, within=NonNegativeReals)
    
    # Resever (down) for each generator
    m.r_down = Var(m.G, m.T, within=NonNegativeReals)
    
    # Wind power output at each node
    m.p_w = Var(m.J, m.T, within=NonNegativeReals)
    
    # Dispatch for each generator above P_MIN
    m.p = Var(m.G, m.T, within=NonNegativeReals)
    
    # Power flow over AC transmission lines
    m.p_ac = Var(m.A, m.T, within=Reals)
    
    # Dummy variables used to compute absolute flows over AC link
    m.p_ac_up = Var(m.A, m.T, within=NonNegativeReals)
    m.p_ac_lo = Var(m.A, m.T, within=NonNegativeReals)

    # Power flow over HVDC links
    def p_hvdc_rule(m, h, t):
        return (-float(df_hvdc.loc[h, 'REVERSE_LIMIT_MW']), float(df_hvdc.loc[h, 'FORWARD_LIMIT_MW']))
    m.p_hvdc = Var(m.H, m.T, bounds=p_hvdc_rule, initialize=0)
    
    # Dummy variables used to compute absolute flows over HVDC links
    m.p_hvdc_up = Var(m.H, m.T, within=NonNegativeReals)
    m.p_hvdc_lo = Var(m.H, m.T, within=NonNegativeReals)
    

    # Expressions
    # -----------
    # Startup cost function
    def C_SU_PRIME_rule(m, g):
        return m.C_SU[g]
    m.C_SU_PRIME = Expression(m.G, rule=C_SU_PRIME_rule)

    # Total power output for each generator
    def p_hat_rule(m, g, t):
        if m.SU_D[g] == 0:
            if t != m.T.last():
                return (m.P_MIN[g] * (m.u[g, t] + m.v[g, t + 1])) + m.p[g, t]
            else:
                return (m.P_MIN[g] * m.u[g, t]) + m.p[g, t]
        else:
            # Startup trajectory
            su_traj = {i + 1: i * (m.P_MIN[g] / m.SU_D[g]) for i in range(0, int(m.SU_D[g]) + 1)}

            # x index
            x_index = [i for i in range(1, int(m.SU_D[g]) + 1) if ((t - i + m.SU_D[g] + 2) <= m.T.last()) and ((t - i + m.SU_D[g] + 2) >= m.T.first())]

            # Shutdown trajectory
            sd_traj = {i + 1: m.P_MIN[g] - (m.P_MIN[g] / m.SD_D[g]) * i for i in range(0, int(m.SD_D[g]) + 1)}

            # y index
            y_index = [i for i in range(2, int(m.SD_D[g]) + 2) if ((t - i + 2) <= m.T.last()) and ((t - i + 2) >= m.T.first())]

            if t != m.T.last():
                return (m.P_MIN[g] * (m.u[g, t] + m.v[g, t + 1]) + m.p[g, t]
                        + sum(su_traj[x] * m.v[g, t - x + int(m.SU_D[g]) + 2] for x in x_index)
                        + sum(sd_traj[y] * m.w[g, t - y + 2] for y in y_index))
            else:
                return ((m.P_MIN[g] * m.u[g, t]) + m.p[g, t]
                        + sum(su_traj[x] * m.v[g, t - x + int(m.SU_D[g]) + 2] for x in x_index)
                        + sum(sd_traj[y] * m.w[g, t - y + 2] for y in y_index))
    m.p_hat = Expression(m.G, m.T, rule=p_hat_rule)

    # Energy output for each generator
    def e_rule(m, g, t):
        if t != m.T.first():
            return ((m.p_hat[g, t - 1] + m.p_hat[g, t]) / 2) * m.DELT
        else:
            return m.p_hat[g, t]
    m.e = Expression(m.G, m.T, rule=e_rule)


    # Constraints
    # -----------
    # Power balance for each NEM zone
    def power_balance_rule(m, j, t):
        gens = [g for g in df_rm.loc[j, 'DUID'] if g in m.G or m.HYDRO]
        if gens:
            return (sum(m.p_hat[g, t] for g in gens if g in m.G) - m.D[j, t] + m.p_w[j, t] + sum(m.HYDRO_OUTPUT[hydro, t] for hydro in gens if hydro in m.HYDRO)
                    == sum(m.p_ac[a, t] * m.AC_C[a, j] for a in m.A) + sum(m.p_hvdc[h, t] * m.HVDC_C[h, j] for h in m.H))
        else:
            return m.D[j, t] == - sum(m.p_ac[a, t] * m.AC_C[a, j] for a in m.A) - sum(m.p_hvdc[h, t] * m.HVDC_C[h, j] for h in m.H)
    m.power_balance = Constraint(m.J, m.T, rule=power_balance_rule)

    # Limit flows over Heywood interconnector
    def heywood_rule(m, t):
        return -500 <= m.p_ac['MEL,SESA', t] <= 600
    m.heywood = Constraint(m.T, rule=heywood_rule)

    # Limit flows over QNI interconnector
    def QNI_rule(m, t):
        return -1078 <= m.p_ac['NNS,SWQ', t] <= 600
    m.QNI = Constraint(m.T, rule=QNI_rule)
    
    # Limit flow over Terranora interconnector
    def terranora_rule(m, t):
        return -210 <= m.p_ac['NNS,SEQ', t] <= 107
    m.terranora = Constraint(m.T, rule=terranora_rule)

    # Limit interconnector flows between VIC and NSW
    def VIC_to_NSW_rule(m, t):
        return -1350 <= sum(m.p_ac[line, t] for line in ['CVIC,SWNSW', 'NVIC,SWNSW', 'NVIC,CAN']) <= 1600
    m.VIC_to_NSW = Constraint(m.T, rule=VIC_to_NSW_rule)

    # DUIDs allocated to each region
    df_region_duids = df_g.reset_index().groupby('NEM_REGION')['DUID'].aggregate(lambda x: set(x))  

    # Ensure upward reserve for each NEM region is maintained
    def reserve_up_rule(m, s, t):
        return sum(m.r_up[g, t] for g in df_region_duids.loc[s] if g in m.G) >= m.D_UP[s]
    m.reserve_up = Constraint(m.S, m.T, rule=reserve_up_rule)

    # Ensure downward reserve for each NEM region is satisfied
    def reserve_down_rule(m, s, t):
        return sum(m.r_down[g, t] for g in df_region_duids.loc[s] if g in m.G) >= m.D_DOWN[s]
    m.reserve_down = Constraint(m.S, m.T, rule=reserve_down_rule)

    # Use SCADA data to initialise 'on' state for first period
    def u0_rule(m, g):
        if df_scada.loc[t_dict[m.T.first()], g] > 1:
            return 1
        else:
            return 0
    m.u0 = Param(m.G, initialize=u0_rule)

    # Logic describing relationship between generator on state, startup state, and shutdown state
    def logic_rule(m, g, t):
        if t != m.T.first():
            return m.u[g, t] - m.u[g, t - 1] == m.v[g, t] - m.w[g, t]
        else:
            return m.u[g, t] - m.u0[g] == m.v[g, t] - m.w[g, t]
    m.logic = Constraint(m.G, m.T, rule=logic_rule)

    # Minimum on time (in time intervals)
    def min_on_time_rule(m, g, t):
        i_index = [i for i in range(t - int(m.TU[g]) + 1, t + 1)]
        if t < m.TU[g]:
            return Constraint.Skip
        else:
            return sum(m.v[g, i] for i in i_index) <= m.u[g, t]
    m.min_on_time = Constraint(m.G, m.T, rule=min_on_time_rule)

    # Minimum off time (in time intervals)
    def min_off_time_rule(m, g, t):
        i_index = [i for i in range(t - int(m.TD[g]) + 1, t + 1)]
        if t < m.TD[g]:
            return Constraint.Skip
        else:
            return sum(m.w[g, i] for i in i_index) <= 1 - m.u[g, t]
    m.min_off_time = Constraint(m.G, m.T, rule=min_off_time_rule)

    # Power output considering upward reserves
    def power_output_reserve_up_rule(m, g, t):
        if t != m.T.last():
            return m.p[g, t] + m.r_up[g, t] <= ((m.P_MAX[g] - m.P_MIN[g]) * m.u[g, t]
                                                - (m.P_MAX[g] - m.SD[g]) * m.w[g, t + 1]
                                                + (m.SU[g] - m.P_MIN[g]) * m.v[g, t + 1])
        else:
            return m.p[g, t] + m.r_up[g, t] <= (m.P_MAX[g] - m.P_MIN[g]) * m.u[g, t]
    m.power_output_reserve_up = Constraint(m.G, m.T, rule=power_output_reserve_up_rule)

    # Power output considering downward reserves
    def power_output_reserve_down_rule(m, g, t):
        return m.p[g, t] - m.r_down[g, t] >= 0
    m.power_output_reserve_down = Constraint(m.G, m.T, rule=power_output_reserve_down_rule)

    # Ramp up limit
    def ramp_up_rule(m, g, t):
        if t == m.T.first():
            return Constraint.Skip
        else:
            return (m.p[g, t] + m.r_up[g, t]) - m.p[g, t - 1] <= m.RU[g]
    m.ramp_up = Constraint(m.G, m.T, rule=ramp_up_rule)

    # Ramp down limit
    def ramp_down_rule(m, g, t):
        if t == m.T.first():
            return Constraint.Skip
        else:
            return -(m.p[g, t] - m.r_down[g, t]) + m.p[g, t - 1] <= m.RD[g]
    m.ramp_down = Constraint(m.G, m.T, rule=ramp_down_rule)

    # Wind output limit (allows curtailment if intermittent generation is too high)
    def wind_output_rule(m, j, t):
        return m.p_w[j, t] <= m.P_W[j, t]
    m.wind_output = Constraint(m.J, m.T, rule=wind_output_rule)
   
    # If shutdown ramp is larger than Pmax, m.p[g,t] + m.r_up[g,t] may be greater than P_MAX[g]
    # Need this constraint to ensure power output is correctly constrained.
    def max_power_output_rule(m, g, t):
        return m.p[g, t] + m.P_MIN[g] + m.r_up[g, t] <= m.P_MAX[g]
    m.max_power_output_rule = Constraint(m.G, m.T, rule=max_power_output_rule)
    
    # Absolute flow over AC links
    def abs_ac_flow_up_rule(m, a, t):
        return m.p_ac_up[a, t] >= m.p_ac[a, t]
    m.abs_ac_flow_up = Constraint(m.A, m.T, rule=abs_ac_flow_up_rule)
    
    def abs_ac_flow_lo_rule(m, a, t):
        return m.p_ac_lo[a, t] >= - m.p_ac[a, t]
    m.abs_ac_flow_lo = Constraint(m.A, m.T, rule=abs_ac_flow_lo_rule)
    
    # Absolute flow over HVDC links
    def abs_hvdc_flow_up_rule(m, h, t):
        return m.p_hvdc_up[h, t] >= m.p_hvdc[h, t]
    m.abs_hvdc_flow_up = Constraint(m.H, m.T, rule=abs_hvdc_flow_up_rule)
    
    def abs_hvdc_flow_lo_rule(m, h, t):
        return m.p_hvdc_lo[h, t] >= - m.p_hvdc[h, t]
    m.abs_hvdc_flow_lo = Constraint(m.H, m.T, rule=abs_hvdc_flow_lo_rule)
    

    # Objective function
    # ------------------
    # Minimise total cost of generation over the time horizon
    def objective_rule(m):
        return sum((m.C_LV[g] * m.e[g, t]) + (m.C_SU_PRIME[g] * m.v[g, t]) for g in m.G for t in m.T) + sum(5 * (m.p_ac_up[a, t] + m.p_ac_lo[a, t]) for a in m.A for t in m.T) + sum(5 * (m.p_hvdc_up[h, t] + m.p_hvdc_lo[h, t]) for h in m.H for t in m.T)
    m.objective = Objective(rule=objective_rule, sense=minimize)


    # Setup solver
    # ------------
    solver = 'gurobi'
    solver_io = 'lp'
    stream_solver = True
    keepfiles = True
    m.dual = Suffix(direction=Suffix.IMPORT)
    opt = SolverFactory(solver, solver_io=solver_io)
    #opt.options['MIPGap'] = 2e-3
    opt.options['TimeLimit'] = 3600

    # Solve model
    results_initial = opt.solve(m, keepfiles=keepfiles, tee=stream_solver)

    # Fix integer variables
    for g in m.G:
        for t in m.T:
            m.u[g, t].fix()
            m.v[g, t].fix()
            m.w[g, t].fix()

    # Re-solve to obtain dual of power balance constraints
    results_final = opt.solve(m, keepfiles=keepfiles, tee=stream_solver)

    # Store all instance solutions in a results object
    m.solutions.store_to(results_final)

    # Retrieve total power and energy output values
    p_hat = []
    e_output = []
    for g in m.G:
        for t in m.T:
            p_hat.append( (g, t, t_dict[t], value(m.p_hat[g, t])) )
            e_output.append( (g, t, t_dict[t], value(m.e[g, t])) )

    # Dataframe for total power output from each unit for each time interval      
    df_p_hat = pd.DataFrame(data=p_hat, columns=['DUID', 'T_INDEX', 'T_STAMP', 'VALUE']).pivot(index='T_STAMP', columns='DUID', values='VALUE')
    
    # Dataframe for total energy output from each unit for each time interval
    df_e_output = pd.DataFrame(data=e_output, columns=['DUID', 'T_INDEX', 'T_STAMP', 'VALUE']).pivot(index='T_STAMP', columns='DUID', values='VALUE')

    # Store dataframes in results dictionary
    results_final['df_p_hat'] = df_p_hat
    results_final['df_e_output'] = df_e_output
    results_final['t_dict'] = t_dict    

    # Save to file
    with open(os.path.join(output_dir, fname), 'wb') as f:
        pickle.dump(results_final, f)
    
    return m

# Model with hydro output determined by UC
m = run_uc_model('uc_results.pickle', fix_hydro=False)

# Model with hydro output fixed to SCADA output values
m_fixed_hydro = run_uc_model('uc_fixed_hydro_results.pickle', fix_hydro=True)


# ## References
# [1] - Morales-España, G., Gentile, C. & Ramos, A. Tight MIP formulations of the power-based unit commitment problem. OR Spectrum 37, 929–950 (2015)
# 
# [2] - Australian Energy Markets Operator. Market Modelling Methodology and Input Assumptions - For Planning the National Electricity Market and Eastern and South-eastern Gas Systems. (AEMO, 2016). at https://www.aemo.com.au/-/media/Files/Electricity/NEM/Planning_and_Forecasting/NTNDP/2016/Dec/Market-Modelling-Methodology-And-Input-Assumptions.pdf
# 
# [3] - Elliston, B., MacGill, I. & Diesendorf, M. Least cost 100% renewable electricity scenarios in the Australian National Electricity Market. Energy Policy 59, 270–282 (2013).
# 
# [4] - Australian Energy Markets Operator. NTNDP Database. (2018). at https://www.aemo.com.au/Electricity/National-Electricity-Market-NEM/Planning-and-forecasting/National-Transmission-Network-Development-Plan/NTNDP-database
