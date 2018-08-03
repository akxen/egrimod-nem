
# coding: utf-8

# # Create tables summarising contents of each dataset

# In[1]:


import os
from decimal import Decimal

import numpy as np
import pandas as pd

pd.set_option('display.max_colwidth', -1)


# ## Paths to directories

# In[2]:


# Network dataset construction directory
network_dir = os.path.join(os.path.curdir, os.path.pardir, '1_network')

# Generator dataset construction directory
generators_dir = os.path.join(os.path.curdir, os.path.pardir, '2_generators')

# Signals dataset construction directory
signals_dir = os.path.join(os.path.curdir, os.path.pardir, '3_load_and_dispatch_signals')

# Output directory
output_dir = os.path.join(os.path.curdir, 'output')


# ## Functions used to parse data

# In[3]:


def get_numerical_range(df, column_name, no_round=False, round_lower=False, round_upper=False, sci_notation_lower=False, sci_notation_upper=False):
    "Round output"
    
    lower = df[column_name].min()
    upper = df[column_name].max()
    
    if no_round:
        return '${0}-{1}$'.format(lower, upper)
    
    else:
        if round_lower:
            lower_out = np.around(lower, decimals=round_lower)

        if round_upper:
            upper_out = np.around(upper, decimals=round_upper)

        if sci_notation_lower:
            lo = df[column_name].min()

            lo_exp = '{:.2e}'.format(lo) # Exponential notation

            # Make latex friendly
            lo_exp = lo_exp.replace('-0','-') 
            lower_out = lo_exp.replace('e', ' \times 10^{') + '}' 

        if sci_notation_upper:
            up = df[column_name].max()

            up_exp = '{:.2e}'.format(up) # Exponential notation

            # Make latex friendly
            up_exp = up_exp.replace('-0','-') 
            upper_out = up_exp.replace('e', ' \times 10^{') + '}'       

        return '${0}-{1}$'.format(lower_out, upper_out)
    
       
def add_caption(table, caption, label):
    "Add caption to table"
    return table.replace('\\end{tabular}\n', '\\end{tabular}\n\\caption{%s}\n\\label{%s}\n' % (caption, label))

def wrap_in_table(table):
    "Wrap tabular in table environment"
    table = table.replace('\\begin{tabular}', '\\begin{table}[H]\n\\begin{tabular}')
    table = table + '\\end{table}'
    return table


def format_table(df_out, caption, label, filename, add_caption_and_label=False):
    "Format table to add caption and labels"

    # Reset index and rename column
    df_out = df_out.reset_index().rename(columns={'index': 'Col. Name'})

    # Format col names so underscores don't cause errors
    df_out['Col. Name'] = df_out['Col. Name'].map(lambda x: x.replace('_', '\_'))
    df_out.index = range(1, len(df_out.index) + 1)
    df_out.index.name = 'Col.'
    df_out = df_out.reset_index()

    # Raw table
    table = df_out.to_latex(escape=False,  index=False, multicolumn=False)

    # Add caption and labels and wrap in table environment
    if add_caption_and_label:
        table_out = add_caption(table, caption=caption, label=label)
        table_out = wrap_in_table(table_out)
    else:
        table_out = table

    # Save to file
    with open(os.path.join(output_dir, filename), 'w') as f:
        f.write(table_out)

    return table_out


# ## Network
# ### Nodes

# In[4]:


def create_network_nodes_table():
    "Create summary of network node datasets"
    
    # Input DataFrame
    df = pd.read_csv(os.path.join(network_dir, 'output', 'network_nodes.csv'))

    # Initialise output DataFrame
    df_out = pd.DataFrame(index=df.columns, columns=['Format', 'Units', 'Range', 'Description'])

    df_out.loc['NODE_ID'] = {'Format': 'str', 'Units': '-', 'Range': get_numerical_range(df, 'NODE_ID', no_round=True), 'Description': 'Node ID'}
    df_out.loc['STATE_NAME'] = {'Format': 'str', 'Units': '-', 'Range': '-', 'Description': 'State in which node is located'}
    df_out.loc['NEM_REGION'] = {'Format': 'str', 'Units': '-', 'Range': '-', 'Description': 'NEM region in which node is located'}
    df_out.loc['NEM_ZONE'] = {'Format': 'str', 'Units': '-', 'Range': '-', 'Description': 'NEM zone in which node is located'}
    df_out.loc['VOLTAGE_KV'] = {'Format': 'int', 'Units': 'kV', 'Range': get_numerical_range(df, 'VOLTAGE_KV', no_round=True), 'Description': 'Node voltage'}
    df_out.loc['RRN'] = {'Format': 'int', 'Units': '-', 'Range': get_numerical_range(df, 'RRN', no_round=True), 'Description': 'If 1 node is a RRN, if 0 node is not a RNN'}
    df_out.loc['PROP_REG_D'] = {'Format': 'float', 'Units': '-', 'Range': get_numerical_range(df, 'PROP_REG_D', round_lower=3, round_upper=3), 'Description': 'Proportion of NEM regional demand consumed at node'}
    df_out.loc['LATITUDE'] = {'Format': 'float', 'Units': 'N$^{\circ}$', 'Range': get_numerical_range(df, 'LATITUDE', round_lower=1, round_upper=1), 'Description': 'Latitude (GDA94)'}
    df_out.loc['LONGITUDE'] = {'Format': 'float', 'Units': 'E$^{\circ}$', 'Range': get_numerical_range(df, 'LONGITUDE', round_lower=1, round_upper=1), 'Description': 'Longitude (GDA94)'}
    
    # Output table after formatting
    table_out = format_table(df_out, caption='Network nodes dataset summary', label='tab: nodes', filename='network_nodes.tex', add_caption_and_label=True)
    
    return table_out

create_network_nodes_table()


# ### AC edges

# In[5]:


def create_network_edges_table():
    "Create table summarising AC network edges dataset"
    
    # Input DataFrame
    df = pd.read_csv(os.path.join(network_dir, 'output', 'network_edges.csv'))

    # Initialise output DataFrame
    df_out = pd.DataFrame(index=df.columns, columns=['Format', 'Units', 'Range', 'Description'])

    df_out.loc['LINE_ID'] = {'Format': 'str', 'Units': '-', 'Range': '-', 'Description': 'Network edge ID'}
    df_out.loc['NAME'] = {'Format': 'str', 'Units': '-', 'Range': '-', 'Description': 'Name of network edge'}
    df_out.loc['FROM_NODE'] = {'Format': 'int', 'Units': '-', 'Range': get_numerical_range(df, 'FROM_NODE', no_round=True), 'Description': 'Node ID for origin node'}
    df_out.loc['TO_NODE'] = {'Format': 'int', 'Units': '-', 'Range': get_numerical_range(df, 'TO_NODE', no_round=True), 'Description': 'Node ID for destination node'}
    df_out.loc['R_PU'] = {'Format': 'float', 'Units': 'p.u.', 'Range': get_numerical_range(df, 'R_PU', sci_notation_lower=True, round_upper=3), 'Description': 'Per-unit resistance'}
    df_out.loc['X_PU'] = {'Format': 'float', 'Units': 'p.u.', 'Range': get_numerical_range(df, 'X_PU', sci_notation_lower=True, round_upper=3), 'Description': 'Per-unit reactance'}
    df_out.loc['B_PU'] = {'Format': 'float', 'Units': 'p.u.', 'Range': get_numerical_range(df, 'B_PU', sci_notation_lower=True, round_upper=3), 'Description': 'Per-unit susceptance'}
    df_out.loc['NUM_LINES'] = {'Format': 'int', 'Units': '-', 'Range': get_numerical_range(df, 'NUM_LINES', no_round=True), 'Description': 'Number of parallel lines'}
    df_out.loc['LENGTH_KM'] = {'Format': 'float', 'Units': 'km', 'Range': get_numerical_range(df, 'LENGTH_KM', round_lower=2, round_upper=1), 'Description': 'Line length'}
    df_out.loc['VOLTAGE_KV'] = {'Format': 'float', 'Units': 'kV', 'Range': get_numerical_range(df, 'VOLTAGE_KV', no_round=True), 'Description': 'Line voltage'}

     # Output table after formatting
    table_out = format_table(df_out, caption='Network edges dataset summary', label='tab: edges', filename='network_edges.tex', add_caption_and_label=True)
    
    return table_out
create_network_edges_table()


# ### HVDC links

# In[6]:


def create_hvdc_links_table():
    "Create summary of HVDC links dataset"

    df = pd.read_csv(os.path.join(network_dir, 'output', 'network_hvdc_links.csv'))

    # Initialise output DataFrame
    df_out = pd.DataFrame(index=df.columns, columns=['Format', 'Units', 'Range', 'Description'])

    df_out.loc['HVDC_LINK_ID'] = {'Format': 'str', 'Units': '-', 'Range': '-', 'Description': 'HVDC link ID'}
    df_out.loc['FROM_NODE'] = {'Format': 'int', 'Units': '-', 'Range': get_numerical_range(df, 'FROM_NODE', no_round=True), 'Description': 'Node ID of origin node'}
    df_out.loc['TO_NODE'] = {'Format': 'int', 'Units': '-', 'Range': get_numerical_range(df, 'TO_NODE', no_round=True), 'Description': 'Node ID of destination node'}
    df_out.loc['FORWARD_LIMIT_MW'] = {'Format': 'float', 'Units': 'MW', 'Range': get_numerical_range(df, 'FORWARD_LIMIT_MW', no_round=True), 'Description': "`From' node to `To' node power-flow limit"}
    df_out.loc['REVERSE_LIMIT_MW'] = {'Format': 'float', 'Units': 'MW', 'Range': get_numerical_range(df, 'REVERSE_LIMIT_MW', no_round=True), 'Description': "`To' node to `From' node power-flow limit"}
    df_out.loc['VOLTAGE_KV'] = {'Format': 'float', 'Units': 'kV', 'Range': get_numerical_range(df, 'VOLTAGE_KV', no_round=True), 'Description': 'HVDC link voltage'}

    # Output table after formatting
    table_out = format_table(df_out, caption='Network HVDC links dataset summary', label='tab: hvdc links', filename='network_hvdc_links.tex', add_caption_and_label=True)
    
    return table_out
create_hvdc_links_table()


# ### AC interconnector links

# In[7]:


def create_ac_interconnector_links_table():
    "Create table summarising AC interconnector connection point locations"

    # Input DataFrame
    df = pd.read_csv(os.path.join(network_dir, 'output', 'network_ac_interconnector_links.csv'))

    # Initialise output DataFrame
    df_out = pd.DataFrame(index=df.columns, columns=['Format', 'Units', 'Range', 'Description'])

    df_out.loc['INTERCONNECTOR_ID'] = {'Format': 'str', 'Units': '-', 'Range': '-', 'Description': 'AC interconnector ID'}
    df_out.loc['FROM_NODE'] = {'Format': 'int', 'Units': '-', 'Range': get_numerical_range(df, 'FROM_NODE', no_round=True), 'Description': 'Node ID of origin node'}
    df_out.loc['FROM_REGION'] = {'Format': 'str', 'Units': '-', 'Range': '-', 'Description': "Region in which `From' node is located"}
    df_out.loc['TO_NODE'] = {'Format': 'int', 'Units': '-', 'Range': get_numerical_range(df, 'TO_NODE', no_round=True), 'Description': 'Node ID for destination node'}
    df_out.loc['TO_REGION'] = {'Format': 'str', 'Units': '-', 'Range': '-', 'Description': "Region in which `To' node is located"}
    df_out.loc['VOLTAGE_KV'] = {'Format': 'float', 'Units': 'kV', 'Range': get_numerical_range(df, 'VOLTAGE_KV', no_round=True), 'Description': 'Line voltage'}

     # Output table after formatting
    table_out = format_table(df_out, caption='AC interconnector locations dataset summary', label='tab: interconnectors - links', filename='network_ac_interconnector_links.tex', add_caption_and_label=True)

    return table_out
create_ac_interconnector_links_table()


# ### Interconnector flow limits

# In[8]:


def create_ac_interconnector_flow_limits_table():
    "Create table summarising interconnector flow limits"
    
    # Input DataFrame
    df = pd.read_csv(os.path.join(network_dir, 'output', 'network_ac_interconnector_flow_limits.csv'))

    # Initialise output DataFrame
    df_out = pd.DataFrame(index=df.columns, columns=['Format', 'Units', 'Range', 'Description'])

    df_out.loc['INTERCONNECTOR_ID'] = {'Format': 'str', 'Units': '-', 'Range': '-', 'Description': 'AC interconnector ID'}
    df_out.loc['FROM_REGION'] = {'Format': 'str', 'Units': '-', 'Range': '-', 'Description': "Region in which `From' node is located"}
    df_out.loc['TO_REGION'] = {'Format': 'str', 'Units': '-', 'Range': '-', 'Description': "Region in which `To' node is located"}
    df_out.loc['FORWARD_LIMIT_MW'] = {'Format': 'float', 'Units': 'MW', 'Range': get_numerical_range(df, 'FORWARD_LIMIT_MW', no_round=True), 'Description': "`From' node to `To' node power-flow limit"}
    df_out.loc['REVERSE_LIMIT_MW'] = {'Format': 'float', 'Units': 'MW', 'Range': get_numerical_range(df, 'REVERSE_LIMIT_MW', no_round=True), 'Description': "`To' node to `From' node power-flow limit"}

    # Output table after formatting
    table_out = format_table(df_out, caption='AC interconnector flow limits summary', label='tab: interconnectors - flow limits', filename='network_ac_interconnector_flow_limits.tex', add_caption_and_label=True)

    return table_out
create_ac_interconnector_flow_limits_table()


# ## Generators

# In[9]:


def create_generators_tables():
    "Create table summarising fields in generator datasets"
    
    # Input DataFrame
    column_dtypes = {'NODE': int, 'REG_CAP': int, 'RR_STARTUP': int, 'RR_SHUTDOWN': int, 'RR_UP': int, 
                     'RR_DOWN': int, 'MIN_ON_TIME': int, 'MIN_OFF_TIME': int, 'SU_COST_COLD': int, 'SU_COST_WARM': int, 
                     'SU_COST_HOT': int}

    df = pd.read_csv(os.path.join(generators_dir, 'output', 'generators.csv'), dtype=column_dtypes)

    # Initialise output DataFrame
    df_out = pd.DataFrame(index=df.columns, columns=['Format', 'Units', 'Range', 'Description'])

    df_out.loc['DUID'] = {'Format': 'str', 'Units': '-', 'Range': '-', 'Description': 'Unique ID for each unit'}
    df_out.loc['STATIONID'] = {'Format': 'str', 'Units': '-', 'Range': '-', 'Description': 'ID of station to which DUID belongs'}
    df_out.loc['STATIONNAME'] = {'Format': 'str', 'Units': '-', 'Range': '-', 'Description': 'Name of station to which DUID belongs'}
    df_out.loc['NEM_REGION'] = {'Format': 'str', 'Units': '-', 'Range': '-', 'Description': 'Region in which DUID is located'}
    df_out.loc['NEM_ZONE'] = {'Format': 'str', 'Units': '-', 'Range': '-', 'Description': 'Zone in which DUID is located'}
    df_out.loc['NODE'] = {'Format': 'int', 'Units': '-', 'Range': get_numerical_range(df, 'NODE', no_round=True), 'Description': 'Node to which DUID is assigned'}
    df_out.loc['FUEL_TYPE'] = {'Format': 'str', 'Units': '-', 'Range': '-', 'Description': 'Primary fuel type'}
    df_out.loc['FUEL_CAT'] = {'Format': 'str', 'Units': '-', 'Range': '-', 'Description': 'Primary fuel category'}
    df_out.loc['EMISSIONS'] = {'Format': 'float', 'Units': 'tCO$_{2}$/MWh', 'Range': get_numerical_range(df, 'EMISSIONS', round_lower=2, round_upper=2), 'Description': 'Equivalent CO$_{2}$ emissions intensity'}
    df_out.loc['SCHEDULE_TYPE'] = {'Format': 'str', 'Units': '-', 'Range': '-', 'Description': 'Schedule type for unit'}
    df_out.loc['REG_CAP'] = {'Format': 'float', 'Units': 'MW', 'Range': get_numerical_range(df, 'REG_CAP', no_round=True), 'Description': 'Registered capacity'}
    df_out.loc['MIN_GEN'] = {'Format': 'float', 'Units': 'MW', 'Range': get_numerical_range(df, 'MIN_GEN', no_round=True), 'Description': 'Minimum dispatchable output'}
    df_out.loc['RR_STARTUP'] = {'Format': 'float', 'Units': 'MW/h', 'Range': get_numerical_range(df, 'RR_STARTUP', no_round=True), 'Description': 'Ramp-rate for start-up'}
    df_out.loc['RR_SHUTDOWN'] = {'Format': 'float', 'Units': 'MW/h', 'Range': get_numerical_range(df, 'RR_SHUTDOWN', no_round=True), 'Description': 'Ramp-rate for shut-down'}
    df_out.loc['RR_UP'] = {'Format': 'float', 'Units': 'MW/h', 'Range': get_numerical_range(df, 'RR_UP', no_round=True), 'Description': 'Ramp-rate up when running'}
    df_out.loc['RR_DOWN'] = {'Format': 'float', 'Units': 'MW/h', 'Range': get_numerical_range(df, 'RR_DOWN', no_round=True), 'Description': 'Ramp-rate down when running'}
    df_out.loc['MIN_ON_TIME'] = {'Format': 'int', 'Units': 'h', 'Range': get_numerical_range(df, 'MIN_ON_TIME', no_round=True), 'Description': 'Minimum on time'}
    df_out.loc['MIN_OFF_TIME'] = {'Format': 'int', 'Units': 'h', 'Range': get_numerical_range(df, 'MIN_OFF_TIME', no_round=True), 'Description': 'Minimum off time'}
    df_out.loc['SU_COST_COLD'] = {'Format': 'int', 'Units': '\$', 'Range': get_numerical_range(df, 'SU_COST_COLD', no_round=True), 'Description': 'Cold start start-up cost'}
    df_out.loc['SU_COST_WARM'] = {'Format': 'int', 'Units': '\$', 'Range': get_numerical_range(df, 'SU_COST_WARM', no_round=True), 'Description': 'Warm start start-up cost'}
    df_out.loc['SU_COST_HOT'] = {'Format': 'int', 'Units': '\$', 'Range': get_numerical_range(df, 'SU_COST_HOT', no_round=True), 'Description': 'Hot start start-up cost'}
    df_out.loc['VOM'] = {'Format': 'float', 'Units': '\$/MWh', 'Range': get_numerical_range(df, 'VOM', no_round=True), 'Description': 'Variable operations and maintenance costs'}
    df_out.loc['HEAT_RATE'] = {'Format': 'float', 'Units': 'GJ/MWh', 'Range': get_numerical_range(df, 'HEAT_RATE', round_lower=1, round_upper=1), 'Description': 'Heat rate'}
    df_out.loc['NL_FUEL_CONS'] = {'Format': 'float', 'Units': '-', 'Range': get_numerical_range(df, 'NL_FUEL_CONS', no_round=True), 'Description': 'No-load fuel consumption as a proportion of full load consumption'}
    df_out.loc['FC_2016-17'] = {'Format': 'float', 'Units': '\$/GJ', 'Range': get_numerical_range(df, 'FC_2016-17', round_lower=1, round_upper=1), 'Description': 'Fuel cost for the year 2016-17'}
    df_out.loc['SRMC_2016-17'] = {'Format': 'float', 'Units': '\$/MWh', 'Range': get_numerical_range(df, 'SRMC_2016-17', round_lower=1, round_upper=1), 'Description': 'Short-run marginal cost for the year 2016-17'}


    # Sources for generator dataset records
    source = dict()
    source['DUID'] = '\cite{aemo_data_2018}'
    source['STATIONID'] = '\cite{aemo_data_2018}'
    source['STATIONNAME'] = '\cite{aemo_data_2018}'
    source['FUEL_TYPE'] = '\cite{aemo_data_2018}'
    source['EMISSIONS'] = '\cite{aemo_current_2018}'
    source['SCHEDULE_TYPE'] = '\cite{aemo_data_2018}'
    source['REG_CAP'] = '\cite{aemo_data_2018}'
    source['MIN_GEN'] = '\cite{aemo_data_2018, aemo_ntndp_2018}'
    source['RR_STARTUP'] = '\cite{aemo_ntndp_2018}'
    source['RR_SHUTDOWN'] = '\cite{aemo_ntndp_2018}'
    source['RR_UP'] = '\cite{aemo_ntndp_2018}'
    source['RR_DOWN'] = '\cite{aemo_ntndp_2018}'
    source['MIN_ON_TIME'] = '\cite{aemo_ntndp_2018}'
    source['MIN_OFF_TIME'] = '\cite{aemo_ntndp_2018}'
    source['SU_COST_COLD'] = '\cite{aemo_ntndp_2018}'
    source['SU_COST_WARM'] = '\cite{aemo_ntndp_2018}'
    source['SU_COST_HOT'] = '\cite{aemo_ntndp_2018}'
    source['VOM'] = '\cite{aemo_ntndp_2018}'
    source['HEAT_RATE'] = '\cite{aemo_ntndp_2018}'
    source['NL_FUEL_CONS'] = '\cite{aemo_ntndp_2018}'
    source['FC_2016-17'] = '\cite{aemo_ntndp_2018}'

    df_out['Source\tnote{$\dagger$}'] = df_out.apply(lambda x: '' if x.name not in source.keys() else source[x.name], axis=1)
    
    # Include double dagger symbol for heatrate record
    df_out = df_out.rename(index={'HEAT_RATE': 'HEAT_RATE\tnote{$\ddagger$}'})

    table_out = format_table(df_out, caption='Generator dataset summary', label='tab: generator dataset', filename='generators.tex')
    
    # Wrap in three part table. Append environments to beginning of tabular
    table_out = '\\begin{table}[H]\n\\begin{threeparttable}\n\\centering\n\\small' + table_out
    
    # Append table notes and environment to end of table
    append_to_end = """\\begin{tablenotes}
    \\item[$\\dagger$] Where no source is given, the value has been derived as part of the dataset construction procedure. NEM\\_REGION and NEM\\_ZONE were found by determining the region and zone of each generator's assigned node. FUEL\\_CAT assigns a generic category to FUEL\\_TYPE. MIN\\_GEN was computed by combining minimum output as a proportion of nameplate capacity from~\cite{aemo_ntndp_2018} with registered capacities from~\\cite{aemo_data_2018}. SRMC\\_2016-17 is calculated from VOM, HEAT\\_RATE, and FC\\_2016-17 fields, using equation~(\\ref{eqn: SRMC calculation}).
    \\item[$\\ddagger$] While not explicitly stated, it is assumed that a lower heating value is referred to. This is consistent with another field in~\\cite{aemo_ntndp_2018} that gives DUID thermal efficiency in terms of lower heating values. 
    \\end{tablenotes}
    \\end{threeparttable}
    \\caption{Generator dataset summary}
    \\label{tab: generator dataset}
    \\end{table}"""
    
    table_out = table_out + append_to_end    
    
    # Save to file
    with open(os.path.join(output_dir, 'generators.tex'), 'w') as f:
        f.write(table_out)
    
    return table_out
create_generators_tables()


# ## Load and dispatch signals
# ### Load signals

# In[10]:


def create_load_signals_table():
    df = pd.read_csv(os.path.join(signals_dir, 'output', 'signals_regional_load.csv'))

    # Initialise output DataFrame
    df_out = pd.DataFrame(index=df.columns, columns=['Format', 'Units', 'Range', 'Description'])

    df_out.loc['SETTLEMENTDATE'] = {'Format': 'timestamp', 'Units': '-', 'Range': '1/6/2017  12:30:00 AM - 1/7/2017  12:00:00 AM', 'Description': 'Trading interval'}
    df_out.loc['NSW1'] = {'Format': 'float', 'Units': 'MW', 'Range': get_numerical_range(df, 'NSW1', round_lower=1, round_upper=1), 'Description': 'New South Wales demand signal'}
    df_out.loc['QLD1'] = {'Format': 'float', 'Units': 'MW', 'Range': get_numerical_range(df, 'QLD1', round_lower=1, round_upper=1), 'Description': 'Queensland demand signal'}
    df_out.loc['SA1'] = {'Format': 'float', 'Units': 'MW', 'Range': get_numerical_range(df, 'SA1', round_lower=1, round_upper=1), 'Description': 'South Australia demand signal'}
    df_out.loc['TAS1'] = {'Format': 'float', 'Units': 'MW', 'Range': get_numerical_range(df, 'TAS1', round_lower=1, round_upper=1), 'Description': 'Tasmania demand signal'}
    df_out.loc['VIC1'] = {'Format': 'float', 'Units': 'MW', 'Range': get_numerical_range(df, 'VIC1', round_lower=1, round_upper=1), 'Description': 'Victoria demand signal'}

    table_out = format_table(df_out, caption='Regional demand signals dataset summary', label='tab: regional demand signals', filename='signals_regional_demand.tex', add_caption_and_label=True)

    return table_out
create_load_signals_table()


# ### Dispatch signals

# In[11]:


df = pd.read_csv(os.path.join(signals_dir, 'output', 'signals_dispatch.csv'))

df_out = pd.DataFrame(columns=['Format', 'Units', 'Range', 'Description'])
df_out.loc['SETTLEMENTDATE'] = {'Format': 'timestamp', 'Units': '-', 'Range': '1/6/2017  12:30:00 AM - 1/7/2017  12:00:00 AM', 'Description': 'Trading interval'}
df_out.loc['(DUID)'] = {'Format': 'float', 'Units': 'MW', 'Range': '-', 'Description': 'DUID dispatch profile'}

# Rename columns
df_out = df_out.reset_index().rename(columns={'index': 'Col. Name'})
df_out.index.name = 'Col.'
df_out = df_out.rename(index={0: '1', 1: '2-{0}'.format(len(df.columns))})
df_out = df_out.reset_index()

# Raw table
table = df_out.to_latex(escape=False, index=False, multicolumn=False)

table_out = add_caption(table, caption='DUID dispatch profiles. Columns correspond to DUIDs.', label='tab: duid dispatch profiles')
table_out = wrap_in_table(table_out)

# Save to file
with open(os.path.join(output_dir, 'signals_dispatch.tex'), 'w') as f:
    f.write(table_out)

