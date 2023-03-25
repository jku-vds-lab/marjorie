from datetime import timedelta, datetime
from math import exp, pow

import numpy as np
import pandas as pd

from helpers import convert_datestrings_df, date_range, construct_colorscale, convert_datestring
from helpers import round_datetime, get_deltas
from variables import target_range, initial_number_of_days

basal_profile_name = 'FirstOne'
basal_profile_name = '/' + basal_profile_name + '/'

def initiate_table(table, data_names, flip=True):
    if 'created_at' in table.columns:
        table = table.rename(columns={'created_at': 'timestamp'})
    table = convert_datestrings_df(table.dropna(), data_names)  # convert date strings to datetime
    if flip:
        table = table[::-1]
    table = table.drop_duplicates().reset_index(drop=True)  # drop duplicates
    return table

# for aaps data
def create_logs_from_aaps_events(events, data_names):
    logs = events[['timestamp', data_names]].dropna()
    logs = logs[::-1]
    return logs

entries = pd.read_csv('datasets/xdrip3.csv', sep=';', parse_dates=[['DAY', 'TIME']], dayfirst=True).iloc[:, :2]
entries.columns = ['timestamp', 'sgv']

events = pd.read_csv('datasets/aaps3.csv', sep=';')
events = events[['Date', 'g', 'U', '%', 'h', 'm']]
events = events.dropna(thresh=2)  # drop rows with all nan
events.loc[:, 'Date'] = events['Date'].apply(convert_datestring)  # convert to datetime
events['m'].fillna(0, inplace=True)
events['Date'] = events['Date'] + events['m'].apply(lambda x: timedelta(minutes=x))
events.columns = ['timestamp', 'carbs', 'bolus', '%', 'h', 'm']

logs_insulin = create_logs_from_aaps_events(events, 'bolus')
logs_carbs = create_logs_from_aaps_events(events, 'carbs')

profile = pd.read_csv('datasets/br.csv')

# initiate tables
logs_sgv = initiate_table(entries, ['sgv'], flip=False)


date_min = min(logs_sgv.timestamp.min(), logs_insulin.timestamp.min(), logs_carbs.timestamp.min())
date_max = max([logs_sgv.timestamp.max(), logs_insulin.timestamp.max(), logs_carbs.timestamp.max()])
start_date = date_max - timedelta(days=initial_number_of_days)
start_date_insights = date_max - timedelta(days=45)
end_date = date_max
days = date_range(date_min, date_max)


# get default basal rate
basal_columns = [i for i in profile.columns if (basal_profile_name + 'basal' in i) and ('timeAsSeconds' not in i)]
timestamps = []
br = []
for i in range(0, len(basal_columns), 2):
    timestamps.append(profile[basal_columns[i]].values[0])
    br.append(profile[basal_columns[i+1]].values[0])

br_default_timestamps = []
br_default_values = []
for day in days:
    for i, timestamp in enumerate(timestamps):
        br_default_timestamps.append(day.replace(hour=int(timestamp[:2]), minute=int(timestamp[3:])))
        br_default_values.append(br[i])
logs_br_default = pd.DataFrame({
    'timestamp': br_default_timestamps + [date_max],
    'br_default': br_default_values + [br_default_values[-1]]
})


# get modified basal rate
df_tbr = events[['timestamp', '%', 'h']]
df_tbr = df_tbr.dropna(thresh=2)
df_tbr.columns = ['timestamp', 'value', 'duration']
df_tbr = initiate_table(df_tbr, ['duration', 'value'])
df_tbr['end_date'] = df_tbr.timestamp + pd.to_timedelta(df_tbr.duration, unit='hours')
df_tbr['end_date_value'] = [logs_br_default.br_default[logs_br_default.timestamp.searchsorted(date)-1] for date in df_tbr.end_date]
logs_br = pd.concat([logs_br_default.copy().rename(columns={'br_default': 'br'}),
                     df_tbr[['timestamp', 'value']].copy().rename(columns={'value': 'br'}),
                     df_tbr[['end_date', 'end_date_value']].copy().rename(columns={'end_date': 'timestamp', 'end_date_value': 'br'})
                     ]).reset_index(drop=True)
logs_br = logs_br.sort_values(by='timestamp')

for i in range(len(df_tbr)):
    logs_br = logs_br.drop(logs_br[(logs_br['timestamp'] > df_tbr['timestamp'].iloc[i]) & (logs_br['timestamp'] < df_tbr['end_date'].iloc[i])].index)

def get_insulin_activity(logs, tp=55, td=360):
    timestamp = logs.timestamp
    insulin = logs.bolus

    def scalable_exp_ia(t, tp, td):
        tp = float(tp)
        td = float(td)
        tau = tp * (1 - tp / td) / (1 - 2 * tp / td)
        a = 2 * (tau / td)
        S = 1 / (1 - a + (1 + a) * exp(-td / tau))
        return (S / pow(tau, 2)) * t * (1 - t / td) * exp(-t / tau)

    number_of_days = len(days)
    insulin_activity = np.zeros((number_of_days + 2) * 1440)  # TODO: find bug
    date_start = datetime.combine(date_min, datetime.min.time())

    x = np.linspace(0, td, num=td)
    y = np.array([scalable_exp_ia(t, tp, td) for t in x])

    for d, i in zip(timestamp, insulin):
        start = int((round_datetime(d) - date_start).total_seconds() / 60)
        insulin_activity[start:start + td] += i * y

    return insulin_activity, date_start

insulin_activity, date_start = get_insulin_activity(logs_insulin)
insulin_activity = insulin_activity[[int((d - date_start).total_seconds() / 60) for d in logs_sgv.timestamp]]
logs_insulin_activity = pd.DataFrame({
    'timestamp': logs_sgv.timestamp,
    'insulin_activity': insulin_activity
})

sgv_delta = get_deltas(logs_sgv['sgv'])
insulin_activity_delta = get_deltas(pd.Series(insulin_activity))

# sgv array for time boxes
tmp = logs_sgv.copy()
tmp['minute'] = tmp.timestamp.dt.hour * 60 + tmp.timestamp.dt.minute
tmp['date'] = tmp.timestamp.dt.date
dates = tmp['date'].unique()
date_dict = dict(enumerate(dates))
inv_date_dict = {v: k for k, v in date_dict.items()}
tmp['line'] = [inv_date_dict[item] for item in tmp['date']]

sgv_array_for_agp = np.full([len(dates), 1440], np.nan)

for i in range(len(tmp)):
    element = tmp.iloc[i]
    sgv_array_for_agp[element['line'], element['minute']] = element['sgv']


# sgv for plots
logs_sgv['low'] = logs_sgv['sgv'] <= target_range[0]
logs_sgv['high'] = logs_sgv['sgv'] >= target_range[1]

threshold_crossings_high = np.diff(logs_sgv.sgv > target_range[1], prepend=False)
logs_sgv_plot = logs_sgv.copy()
logs_sgv_plot.loc[threshold_crossings_high, 'sgv'] = target_range[1]

threshold_crossings_low = np.diff(logs_sgv.sgv < target_range[0], prepend=False)
logs_sgv_plot.loc[threshold_crossings_low, 'sgv'] = target_range[0]

logs_sgv_plot['low'] = logs_sgv_plot['sgv'] <= target_range[0]
logs_sgv_plot['high'] = logs_sgv_plot['sgv'] >= target_range[1]

# targets

targets = [1, 54, 70, 180, 250, max(logs_sgv.sgv)]

colors = [
        'rgb(251,90,82)',
        'rgb(255,140,126)',
        'rgb(120,211,168)',
        'rgb(188,155,233)',
        'rgb(139,99,213)'
]
domain = [target/targets[-1] for target in targets]

colors_insulin = [
    'rgb(201,231,246)',
    'rgb(173,225,246)',
    'rgb(146,220,245)',
    'rgb(118,214,245)',
    'rgb(90,209,245)'
]

colors_carbs = [
    'rgb(234,231,220)',
    'rgb(239,227,193)',
    'rgb(245,222,167)',
    'rgb(250,218,140)',
    'rgb(255,213,114)'
]

colorscales = [construct_colorscale(colors_insulin),
               construct_colorscale(colors_carbs),
               construct_colorscale(colors, domain)]

colorscale = construct_colorscale(colors, domain)

colors_categorical = [
    # 'rgb(229,236,246)',
    'rgb(251,90,82)',
    'rgb(120,211,168)',
    'rgb(139,99,213)',
    # 'rgb(139,99,213)'
]

colorscale_categorical = construct_colorscale(colors_categorical)