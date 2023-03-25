from dateutil.relativedelta import relativedelta
import datetime as dt
from datetime import timedelta, date

import numpy as np
import pandas as pd
from dateutil import parser
from dateutil.relativedelta import relativedelta

from variables import target_range, target_range_extended


def convert_datestring(datestring):
    return parser.parse(datestring, ignoretz=True)


def convert_datestrings_df(df, data_names):
    data = {k: [] for k in ['timestamp'] + data_names}

    for i in range(len(df['timestamp'])):
        datestring = str(df['timestamp'].iloc[i])
        try:
            data['timestamp'].append(convert_datestring(datestring))
            for data_name in data_names:
                entry = df[data_name].iloc[i]
                data[data_name].append(entry)
        except Exception as e:
            print(datestring)
        else:
            continue
    return pd.DataFrame(data)


def round_datetime(d):
    td = dt.timedelta(hours=d.hour, minutes=d.minute, seconds=d.second, microseconds=d.microsecond)
    to_min = dt.timedelta(minutes=round(td.total_seconds() / 60))
    return dt.datetime.combine(d, dt.time(0)) + to_min


def normalize(time_data):
    time_data = (time_data - np.min(time_data)) / (np.max(time_data) - np.min(time_data))
    return time_data


def get_deltas(array):
    delta = array.diff()
    delta[0] = delta[1]
    return delta


def get_timedelta(group):
    delta = {
        'day': timedelta(days=1),
        'week': timedelta(days=7),
        'month': relativedelta(months=1),
        'quarter': relativedelta(months=3)
    }
    return delta[group]


def date_range(date_min, date_max):
    delta = date_max - date_min
    days = [date_min + timedelta(days=i) for i in range(delta.days + 1)]
    return days


def get_start_date_from_zoom(group, zoom_start_date):
    current_quarter = round((zoom_start_date.month - 1) / 3 + 1)
    start_date = {
        'day': zoom_start_date.replace(hour=0, minute=0, second=0, microsecond=0),
        'week': zoom_start_date - timedelta(days=zoom_start_date.weekday()),
        'month': (zoom_start_date.replace(day=1) - timedelta(days=1)).replace(day=1),
        'quarter': date(zoom_start_date.year, 3 * ((zoom_start_date.month - 1) // 3) + 1, 1)
    }
    return start_date[group]


def dt_details_function(group):
    dt_details = {
        'day': lambda x: str(x.date()),
        'week': lambda x: (int(x.strftime("%V")), x.year),
        'month': lambda x: (x.month, x.year),
        'quarter': lambda x: (x.quarter, x.year),
    }
    return dt_details[group]


def get_infos_from_group(group):
    values = {
        'day': [0.5, 6, 12, 18, 24],
        'week': list(range(7)),
        'month': list(range(12)),
        'quarter': list(range(1, 5)),
    }

    labels = {
        'day': ['0 am', '6 am', '12 am', '6 pm', '0 pm'],
        'week': ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'],
        'month': ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'],
        'quarter': ['Q1', 'Q2', 'Q3', 'Q4']
    }
    return values[group], labels[group]


def construct_colorscale(colors, domain=None):
    if not domain:
        domain = np.linspace(10 ** (-5), 1, num=len(colors) + 1)
    domain = np.insert(domain, 0, 0)
    colors = ['rgba(248,249,250,1)'] + colors

    colorscale = []
    for i, color in enumerate(colors):
        colorscale.extend(
            [[domain[i], color],
             [domain[i + 1], color]]
        )
    # print(colorscale)
    return colorscale


def highlight_data(x, y, y_0):
    x = np.repeat(x, 2)
    if len(y) == 2:
        y = [y[0], y[1], y[1], y[0]]
    else:
        y = np.repeat(y, 2)
        y = np.concatenate([[y_0], y])
        y = y[:-1]
        y[-1] = y_0
    return x, y


def color_range(percentage, initial_color=(230, 234, 238), target_color=(255, 213, 114)):
    result_color = tuple(int(c1 + percentage * (c2 - c1))
                         for c1, c2 in zip(initial_color, target_color))
    return 'rgb({}, {}, {})'.format(result_color[0], result_color[1], result_color[2])


def get_tir(sgv):
    def time_in_range(sgv):
        in_range = np.where((sgv >= target_range[0]) & (sgv <= target_range[1]))[0]
        tir = len(in_range) / len(sgv)
        return tir

    def time_very_low(sgv):
        under_range = np.where(sgv < target_range_extended[0])[0]
        tur = len(under_range) / len(sgv)
        return tur

    def time_low(sgv):
        under_range = np.where((sgv >= target_range_extended[0]) & (sgv < target_range_extended[1]))[0]
        tur = len(under_range) / len(sgv)
        return tur

    def time_high(sgv):
        above_range = np.where((sgv > target_range_extended[-2]) & (sgv <= target_range_extended[-1]))[0]
        tar = len(above_range) / len(sgv)
        return tar

    def time_very_high(sgv):
        above_range = np.where(sgv > target_range_extended[-1])[0]
        tar = len(above_range) / len(sgv)
        return tar


    tir = [time_very_low(sgv),
           time_low(sgv),
           time_in_range(sgv),
           time_high(sgv),
           time_very_high(sgv)
           ]

    tir = [int(round(item * 100)) for item in tir]

    return tir


def calculate_tir_time(tir):
    min = int(tir/100 * 24 * 60)
    hours = int(min / 60)
    minutes = min % 60
    if hours == 0:
        hours = '0h'
    else:
        hours = str(hours) + 'h'
    if minutes == 0:
        minutes = '0m'
    else:
        minutes = str(minutes) + 'm'
    time = hours + ' ' + minutes
    return time


def get_statistics(sgv):
    def time_in_range(sgv):
        in_range = np.where((sgv >= target_range[0]) & (sgv <= target_range[1]))[0]
        tir = len(in_range) / len(sgv)
        return tir

    def time_very_low(sgv):
        under_range = np.where(sgv < target_range_extended[0])[0]
        tur = len(under_range) / len(sgv)
        return tur

    def time_low(sgv):
        under_range = np.where((sgv >= target_range_extended[0]) & (sgv < target_range_extended[1]))[0]
        tur = len(under_range) / len(sgv)
        return tur

    def time_high(sgv):
        above_range = np.where((sgv > target_range_extended[-2]) & (sgv <= target_range_extended[-1]))[0]
        tar = len(above_range) / len(sgv)
        return tar

    def time_very_high(sgv):
        above_range = np.where(sgv > target_range_extended[-1])[0]
        tar = len(above_range) / len(sgv)
        return tar

    def sgv_mean(sgv):
        return sgv.mean()

    def sgv_std(sgv):
        return sgv.std()

    def sgv_ea1c(sgv):
        return (sgv.mean() + 46.7) / 28.7

    return {
        'mean': sgv_mean(sgv),
        'std': sgv_std(sgv),
        'ea1c': sgv_ea1c(sgv),
        'time_very_low': time_very_low(sgv),
        'time_low': time_low(sgv),
        'time_in_range': time_in_range(sgv),
        'time_high': time_high(sgv),
        'time_very_high': time_very_high(sgv)

    }


def datestr(date_dt):
    return date_dt.strftime('%Y.%m.%dT%H:%M')


def get_df_of_date(df, day):
    mask = (df['timestamp'].dt.date == day)
    df_day = df[mask]
    return df_day


def get_df_between_dates(df, start_date, end_date, weekday_filter=None):
    mask = (df['timestamp'] > start_date) & (df['timestamp'] <= end_date)
    df = df[mask]
    if weekday_filter is not None:
        df['weekday'] = df.timestamp.dt.weekday
        df = df[df.weekday.isin(weekday_filter)]

    return df


def get_mean_per_day(logs, log_type):
    logs['date'] = logs.timestamp.dt.date
    mean_value = round(logs.groupby('date').agg({log_type: 'sum'}).mean(), 1)
    return mean_value


def check_timebox(array, y_range):
    array = np.ma.array(array, mask=np.isnan(array))
    in_selected_range = (array > y_range[0]) & (array < y_range[1])
    result = np.where((np.sum(in_selected_range, axis=1) / array.count(axis=1)) >= 0.4)[0]
    return result


def get_log_indices(logs, dates):
    indices = logs.timestamp.searchsorted(dates)
    return indices
