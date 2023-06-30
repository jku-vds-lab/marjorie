from collections import Counter
from datetime import timedelta

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import scipy
import scipy.spatial.distance as ssd
from dtaidistance import dtw
from plotly.subplots import make_subplots
from scipy.cluster.hierarchy import single, complete, average, ward, fcluster
from scipy.signal import savgol_filter
from tslearn.utils import to_time_series_dataset

from colors import colors, colors_agp, colors_patterns, colors_pattern_curves, colors_heatmap, domain_heatmap
from helpers import get_df_between_dates, get_log_indices, construct_colorscale
from overview import hierarchy_cluster
from preprocessing import logs_sgv, logs_carbs, logs_insulin, start_date_insights, end_date
from variables import target_range, font, num_insight_patterns
from variables import time_before_meal, time_after_meal, times_of_day, time_before_hypo, time_after_hypo


def hierarchical_clustering(distance_matrix, method='average'):
    distance_matrix = ssd.squareform(distance_matrix)  # condense distance matrix
    if method == 'complete':
        Z = complete(distance_matrix)
    if method == 'single':
        Z = single(distance_matrix)
    if method == 'average':
        Z = average(distance_matrix)
    if method == 'ward':
        Z = ward(distance_matrix)
    return Z


def get_clusters_from_z(Z, max_d=200):
    clusters = fcluster(Z, t=max_d, criterion='distance')
    return clusters


def get_dtw_distance(data):
    distance_matrix = dtw.distance_matrix_fast([np.array(item) for item in data])
    return distance_matrix


def sort_clusters(clusters):
    frequencies = Counter(clusters)
    frequencies = dict(frequencies.most_common())
    clusters_ordered = [list(frequencies.keys()).index(n)+1 for n in clusters]
    return np.array(clusters_ordered)


def get_insight_clusters(dataset, max_d=200):
    distance_matrix = get_dtw_distance(dataset)
    Z = hierarchical_clustering(distance_matrix)
    clusters = get_clusters_from_z(Z, max_d)
    clusters = sort_clusters(clusters)
    return clusters


def get_insight_dataset(start_date_indices, end_date_indices, cluster=True):
    time_series = [logs_sgv.sgv.iloc[start_idx:end_idx].tolist() for start_idx, end_idx in zip(start_date_indices, end_date_indices)]
    dataset = list(filter(lambda x: x, time_series))
    if not cluster:
        dataset = to_time_series_dataset(dataset)
    return dataset


def get_logs_from_indices(indices):
    logs_indices = {}
    for logs, log_type in zip([logs_sgv, logs_carbs, logs_insulin], ['sgv', 'carbs', 'insulin']):
        start_date_indices = indices[log_type][0]
        end_date_indices = indices[log_type][1]
        logs_indices[log_type] = [logs.iloc[start_idx:end_idx] for start_idx, end_idx in zip(start_date_indices, end_date_indices)]
    # remove emtpy entries
    empties = [i for i, item in enumerate(logs_indices['sgv']) if item.empty]
    for log_type in ['sgv', 'carbs', 'insulin']:
        logs_indices[log_type] = [item for i, item in enumerate(logs_indices[log_type]) if i not in empties]
    return logs_indices


def get_min_max(array):
    array_min = np.nanmin(array, axis=0)
    array_max = np.nanmax(array, axis=0)
    return array_min, array_max


def get_average(array):
    med = np.nanmedian(array, axis=0)
    min = np.nanmin(array, axis=0)
    avg = med.copy()
    avg[min < 80] = np.mean(np.array([med[min < 80], min[min < 80]]), axis=0)
    return avg


def dropna(x):
    x = x[~np.isnan(x)]
    return x


def draw_hierarchical_pattern_overview(dataset, dataset_array, dataset_carbs, dataset_bolus, text, time_before=0):
    fig = make_subplots(rows=4, cols=1, shared_xaxes=True, row_heights=[0.82, 0.06, 0.06, 0.06], vertical_spacing=0.025)

    for data_sgv, data_carbs, data_bolus in zip(dataset, dataset_carbs, dataset_bolus):

        x = np.linspace(time_before * (-1), len(data_sgv) * 5 / 60 - time_before, len(data_sgv))

        fig.add_trace(
            go.Scatter(
                x=x,
                y=data_sgv.sgv,
                # hovertext=[data_sgv.timestamp.iloc[0].date() for i in range(len(data_sgv))],
                hovertext=data_sgv.timestamp.iloc[0].strftime('%a') + ', ' + data_sgv.timestamp.iloc[0].strftime('%y-%m-%d') + ', ' + data_sgv.timestamp.iloc[0].strftime('%H:%M') + '-' + data_sgv.timestamp.iloc[
                    -1].strftime('%H:%M'),
                hoverinfo='text',
                mode='lines',
                connectgaps=False,
                showlegend=False,
                line=dict(color=colors['highlight']),
            ),
            row=1, col=1
        )

        if not data_carbs.empty:
            data_carbs['x'] = data_carbs.timestamp - data_sgv.timestamp.iloc[0]
            data_carbs['x'] = data_carbs.x.dt.seconds / 3600 - time_before
            data_carbs['alphas'] = data_carbs.carbs / 70 * 0.8
            data_carbs['alphas'][data_carbs['alphas'] > 1] = 1
            for i in range(len(data_carbs)):
                fig.add_trace(
                    go.Scatter(
                        x=[data_carbs.x.iloc[i]],
                        y=[0],
                        mode='markers',
                        hovertext=get_hover_data(data_carbs, 'carbs', 'g', 0),
                        hoverinfo='text',
                        marker=dict(
                            color='rgba(255,213,114, {})'.format(data_carbs.alphas.iloc[i]),
                            size=10
                        )
                    ),
                    row=3, col=1
                )

        if not data_bolus.empty:
            data_bolus['x'] = data_bolus.timestamp - data_sgv.timestamp.iloc[0]
            data_bolus['x'] = data_bolus.x.dt.seconds / 3600 - time_before
            data_bolus['alphas'] = data_bolus.bolus / 10 * 0.8
            data_bolus['alphas'][data_bolus['alphas'] > 1] = 1
            for i in range(len(data_bolus)):
                fig.add_trace(
                    go.Scatter(
                        x=[data_bolus.x.iloc[i]],
                        y=[0],
                        hovertext=get_hover_data(data_bolus, 'bolus', 'U', 1),
                        hoverinfo='text',
                        mode='markers',
                        marker=dict(
                            color='rgba(90,209,245, {})'.format(data_bolus.alphas.iloc[i]),
                            size=10
                        )
                    ),
                    row=4, col=1
                )

    array = dataset_array.copy()
    array_min, array_max = get_min_max(array)
    array_min = dropna(array_min)
    array_max = dropna(array_max)
    x = np.linspace(time_before * (-1), len(array_min) * 5 / 60 - time_before, len(array_min))

    # low
    low_min = array_min.copy()
    low_min[array_min >= 70] = 70
    low_max = array_max.copy()
    low_max[array_max >= 70] = 70

    # high
    high_min = array_min.copy()
    high_min[array_min <= 180] = 180
    high_max = array_max.copy()
    high_max[array_max <= 180] = 180

    # normal
    normal_min = array_min.copy()
    normal_min[array_min <= 70] = 70
    normal_min[array_min >= 180] = 180
    normal_max = array_max.copy()
    normal_max[array_max <= 70] = 70
    normal_max[array_max >= 180] = 180

    fig.add_trace(
        go.Scatter(
            x=np.append(x, np.flip(x)),
            y=np.append(low_max, np.flip(low_min)),
            fill='toself',
            fillcolor='rgba(255,140,126,0.5)',  # colors['bg_low']
            hoverinfo='skip',
            line=dict(color='rgba(0,0,0,0)'),
            showlegend=False,
        ),
        row=1, col=1
    )

    fig.add_trace(
        go.Scatter(
            x=np.append(x, np.flip(x)),
            y=np.append(high_max, np.flip(high_min)),
            fill='toself',
            fillcolor='rgba(188,155,233, 0.5)',  # colors['bg_high'],
            hoverinfo='skip',
            line=dict(color='rgba(0,0,0,0)'),
            showlegend=False,
        ),
        row=1, col=1
    )

    fig.add_trace(
        go.Scatter(
            x=np.append(x, np.flip(x)),
            y=np.append(normal_max, np.flip(normal_min)),
            fill='toself',
            fillcolor='rgba(120,211,168, 0.5)',  # colors['bg_target'],
            hoverinfo='skip',
            line=dict(color='rgba(0,0,0,0)'),
            showlegend=False,
        ),
        row=1, col=1
    )

    tickvals = np.linspace(time_before * (-1), round(len(array_min) * 5 / 60 - time_before), round(len(array_min) * 5 / 60) + 1)
    ticklabels = [item if item != 0 else text for item in tickvals.tolist()]

    avg_array = dropna(get_average(array))
    z = [avg_array[0].item()] + [avg_array[i].item() for i in range(11, len(avg_array), 12)]
    text = [str(round(item)) for item in z]

    colormap_heatmap = construct_colorscale(colors_heatmap, domain_heatmap)

    # average glucose
    fig.add_trace(
        go.Heatmap(
            z=[z],
            x=tickvals,
            y=[''],
            hovertext=[z],
            hoverinfo='text',
            # text=[text],
            # texttemplate="%{text}",
            colorscale=colormap_heatmap,
            zmin=1,
            zmax=400,
            showscale=False
        ),
        row=2, col=1
    )

    # vertical line
    fig.add_vline(x=0, line_dash="dash")

    fig.update_layout(
        plot_bgcolor=colors['background'],
        xaxis4=dict(
            tickvals=tickvals,
            ticktext=ticklabels,
            title='<b>Hours</b>',
            title_font_size=10,
            range=[tickvals[0], tickvals[-1]]
        ),
        yaxis=dict(
            range=[40,400],
            tickfont=dict(size=8)
        ),
        yaxis3=dict(
            showticklabels=False,
            range=[-0.5, 0.5],
            showgrid=False
        ),
        yaxis4=dict(
            showticklabels=False,
            range=[-0.5, 0.5],
            showgrid=False
        ),
        width=500, height=300, margin=dict(l=10, r=10, t=25, b=30),
        showlegend=False,
    )
    return fig


def draw_pattern_overview_curves(dataset):
    layout = go.Layout(width=620, height=200, margin=dict(l=10, r=10, t=25, b=10))
    fig = go.Figure(layout=layout, layout_yaxis_range=[40, 400])
    time_series = dataset[:, :, 0]
    for i in range(time_series.shape[0]):
        fig.add_trace(go.Scatter(
            x=list(range(len(time_series[i]))),
            y=time_series[i],
            mode='lines',
            connectgaps=False,
            showlegend=False,
            line=dict(color=colors['bg_target'])
        ))

    fig.update_layout(
        width=460, height=200,
        xaxis_range=(0, time_series.shape[1]),
        showlegend=True,
        legend=dict(
            font=dict(
                size=10,
            ),
            itemwidth=30
        ),
        plot_bgcolor=colors['background'],
    )
    fig.update_layout(dragmode="select", clickmode='event+select')
    return fig


def draw_pattern_overview(dataset, clusters, n_clusters):
    layout = go.Layout(width=620, height=200, margin=dict(l=10, r=10, t=25, b=10))
    fig = go.Figure(layout=layout, layout_yaxis_range=[0, 400])

    time_series = dataset[:, :, 0]

    for i in range(time_series.shape[0]):
        fig.add_trace(go.Scatter(
            x=list(range(len(time_series[i]))),
            y=time_series[i],
            mode='lines',
            connectgaps=False,
            showlegend=False,
            line=dict(color=colors_pattern_curves[clusters[i]]),
            legendgroup='group_{}'.format(clusters[i]),
        ))

    for i in range(n_clusters):
        array = time_series[clusters == i + 1]
        array_min, array_max = get_min_max(array)
        # print(array_min, array_max)
        array_min = dropna(array_min)
        array_max = dropna(array_max)
        array_min = savgol_filter(array_min, 11, 3) - 5
        array_max = savgol_filter(array_max, 11, 3) + 5
        x = list(range(len(array_min)))
        fig.add_trace(
            go.Scatter(
                x=np.append(x, np.flip(x)),
                y=np.append(array_max, np.flip(array_min)),
                customdata=np.ones(2 * len(x)) * i,
                fillcolor=colors_patterns[i],
                fill='toself',
                hoverinfo='skip',
                line=dict(color='rgba(255,255,255, 0)'),
                name='TIR: 70%<br>TUR: 5%',
                legendgroup='group_{}'.format(i),
                legendgrouptitle_text='Pattern {}:'.format(i + 1)
            )
        )
        fig.update_layout(
            width=460, height=200,
            xaxis_range=(0, time_series.shape[1]),
            legend=dict(
                font=dict(
                    size=10,
                ),
                itemwidth=30
            ),
            legend_grouptitlefont=dict(size=11),
            legend_itemclick='toggleothers',
            plot_bgcolor=colors['background'],
        )
        fig.update_layout(dragmode="select", clickmode='event+select')

    return fig


def add_time_before_after(start_dates, hours_before, hours_after):
    end_dates = pd.Series(start_dates) + pd.Series([timedelta(hours=hours_after)] * len(start_dates))
    start_dates = pd.Series(start_dates) - pd.Series([timedelta(hours=hours_before)] * len(start_dates))
    return start_dates, end_dates


def find_hypo_periods():
    # index_pos_list = [ i for i in range(len(logs_sgv.sgv)) if logs_sgv.sgv[i]<70 ]
    is_hypo = logs_sgv.sgv < 70
    regions = scipy.ndimage.find_objects(scipy.ndimage.label(is_hypo)[0])
    regions = [item for item in regions if item[0].start != item[0].stop]
    beginnings = [logs_sgv.timestamp[r].iloc[0] for r in regions]
    clusters_timestamps = [logs_sgv.timestamp[r].to_list() for r in regions]
    clusters_sgv = [logs_sgv.sgv[r].to_list() for r in regions]

    time_differences = np.diff(beginnings)
    consecutive = time_differences < timedelta(minutes=21)
    idx = np.where(consecutive)[0]
    timestamps = []
    sgv = []
    start_dates = []
    end_dates = []
    for i in range(len(clusters_timestamps) - 1):
        if i not in idx + 1:
            start_dates.append(clusters_timestamps[i][0])
            if i in idx:
                timestamps.append(clusters_timestamps[i] + clusters_timestamps[i + 1])
                sgv.append(clusters_sgv[i] + clusters_sgv[i + 1])
                end_dates.append(clusters_timestamps[i + 1][-1])
            else:
                timestamps.append(clusters_timestamps[i])
                sgv.append(clusters_sgv[i])
                end_dates.append(clusters_timestamps[i][-1])
    return start_dates, end_dates


def draw_pattern_detail_plot(day, sgv_today, carbs_today, insulin_today, x_range):
    fig_timeline = make_subplots(rows=1, cols=1, specs=[[{"secondary_y": True}]])

    # sgv
    y_sgv = sgv_today.sgv.fillna(0)
    fig_timeline.add_trace(
        go.Scatter(
            x=sgv_today.timestamp,
            y=y_sgv,
            mode='lines',
            line=dict(color=colors['bg_target']),
            connectgaps=False,
            hovertext=sgv_today.sgv,
            hoverinfo='text'
        ),
    )

    # below range
    sgv_low = sgv_today.loc[sgv_today['low']]
    if len(sgv_low) > 1:
        fig_timeline.add_trace(
            go.Scatter(
                x=sgv_low.timestamp.to_list() + [sgv_low.timestamp.iloc[-1], sgv_low.timestamp.iloc[0]],
                y=sgv_low.sgv.fillna(0).to_list() + [target_range[0], target_range[0]],
                fill='toself',
                fillcolor=colors_agp['under_range_90th'],
                mode='lines',
                line=dict(color=colors['bg_low']),
                connectgaps=False,
                hoverinfo='skip'
            ),
        )

    # above range
    sgv_high = sgv_today.loc[sgv_today['high']]
    fig_timeline.add_trace(
        go.Scatter(
            x=sgv_high.timestamp.to_list() + [sgv_high.timestamp.iloc[-1], sgv_high.timestamp.iloc[0]],
            y=sgv_high.sgv.fillna(0).to_list() + [target_range[1], target_range[1]],
            fill='toself',
            fillcolor=colors_agp['above_range_90th'],
            mode='lines',
            line=dict(color=colors['bg_high']),
            connectgaps=False,
            hoverinfo='skip'
        ),
    )

    # target
    fig_timeline.add_trace(
        go.Scatter(
            x=sgv_today.timestamp,
            y=[target_range[1]] * len(sgv_today),
            mode='lines',
            line=dict(color='white'),
            connectgaps=False,
            hoverinfo='skip'
        ),
    )

    fig_timeline.add_trace(
        go.Scatter(
            x=sgv_today.timestamp,
            y=[target_range[0]] * len(sgv_today),
            mode='lines',
            line=dict(color='white'),
            connectgaps=False,
            hoverinfo='skip'
        ),
    )

    # carbs
    fig_timeline.add_trace(
        go.Bar(
            x=carbs_today.timestamp,
            y=carbs_today.carbs,
            marker=dict(opacity=0.4),
            marker_color=colors['carbs'],
            yaxis='y2',
            width=8 * 3600 * 24,
        ),
    )

    # insulin
    fig_timeline.add_trace(
        go.Bar(
            x=insulin_today.timestamp,
            y=insulin_today.bolus,
            marker=dict(opacity=0.4),
            marker_color=colors['bolus'],
            yaxis='y3',
            width=8 * 3600 * 24,
        ),
    )

    fig_timeline.update_xaxes(type="date", autorange=False, range=x_range)
    fig_timeline.update_layout(xaxis_rangeslider_visible=False,
                               xaxis_type="date",
                               showlegend=False,
                               width=630, height=150,
                               margin=dict(t=25, b=10, l=0, r=0),
                               plot_bgcolor='rgba(248,249,250,1)',
                               xaxis=dict(visible=False, showgrid=True),
                               yaxis=dict(
                                   range=[0, 400],
                                   tickfont_size=8,
                                   visible=False
                               ),
                               yaxis2=dict(
                                   range=[0, max(logs_carbs.carbs)],
                                   tickfont_size=8,
                                   overlaying="y",
                                   showgrid=False,
                                   visible=False
                               ),
                               yaxis3=dict(
                                   range=[0, max(logs_insulin.bolus)],
                                   tickfont_size=8,
                                   overlaying="y",
                                   showgrid=False,
                                   visible=False
                               ),

                               font=dict(
                                   family=font,
                               ),
                               )

    return fig_timeline


def get_hover_data(agg_data, log_type, unit, round_):
    # hover data
    dates = agg_data.timestamp
    amount = agg_data[log_type]

    name = dates.dt.strftime('%a') + ', ' + dates.dt.strftime('%y-%m-%d') + ', ' + dates.dt.strftime('%H:%M') + ': ' + '<b>' + amount.round(round_).astype(str) + ' {}</b>'.format(unit)
    return name


def get_logs_meals(start_date, end_date, time_before, time_after):
    carbs_df = get_df_between_dates(logs_carbs, start_date, end_date)
    agg_data = hierarchy_cluster(carbs_df)
    agg_data = agg_data[agg_data[('carbs', 'sum')] >= 40]
    agg_data['start_date'] = agg_data[('timestamp', 'min')] - timedelta(hours=time_before)
    agg_data['end_date'] = agg_data[('timestamp', 'min')] + timedelta(hours=time_after)
    min_hour = agg_data.min_time.dt.hour
    agg_data['time_of_day'] = min_hour.map(times_of_day)

    return agg_data


def get_logs_hypos(start_date, end_date, time_before, time_after):
    def blocks_with_tolerance(w, n=3, m=3):
        # find blocks with at least n consecutive true values with a tolerance of m false values in between
        b = w.ne(w.shift()).cumsum() * w
        y = b.map(b.mask(b == 0).value_counts()) <= n

        w1 = ~y
        b1 = w1.ne(w1.shift()).cumsum() * w1
        y1 = b1.map(b1.mask(b1 == 0).value_counts()) == m

        y = y | y1
        return y

    def get_start_end_indices(df):
        v = (df != df.shift()).cumsum()
        u = df.groupby(v).agg(['all', 'count'])
        m = u['all'] & u['count'].ge(3)
        indices = pd.DataFrame(list(df.groupby(v).apply(lambda x: [x.index[0], x.index[-1]])[m]))
        return indices

    df_logs = get_df_between_dates(logs_sgv, start_date - timedelta(days=60), end_date)
    df = blocks_with_tolerance(df_logs['sgv'] <= 70)
    indices = get_start_end_indices(df)
    hypo_starts = df_logs.timestamp.loc[indices[0]]
    logs_hypos = pd.DataFrame(list(hypo_starts - timedelta(hours=time_before)), columns=['start_date'])
    logs_hypos['end_date'] = list(hypo_starts + timedelta(hours=time_after))
    min_hour = logs_hypos.start_date.dt.hour
    logs_hypos['time_of_day'] = min_hour.map(times_of_day)
    hypo_starts = logs_hypos.start_date
    return logs_hypos, hypo_starts


def hierarchy_cluster(logs_today, log_type='carbs', max_d=1.5):
    logs_today['time'] = logs_today.timestamp.values.astype(np.int64) // 10 ** 9 // 60 // 60
    data = logs_today.time.to_numpy()
    nnumbers = data.reshape(-1)
    data = data.reshape(-1, 1)
    Z = scipy.cluster.hierarchy.ward(data)
    color = fcluster(Z, t=max_d, criterion='distance')
    grouped_item = pd.DataFrame(list(zip(nnumbers, color, logs_today[log_type], logs_today.timestamp)), columns=['numbers', 'segment', log_type, 'timestamp']).groupby('segment')
    agg_data = grouped_item.agg({log_type: [sum, list], 'timestamp': [min, max, list]})
    agg_data = agg_data.reset_index()
    agg_data['timedelta'] = agg_data.timestamp['max'] - agg_data.timestamp['min']

    agg_data['middle_time'] = agg_data.timestamp['min'] + agg_data['timedelta'] / 2
    agg_data['min_time'] = agg_data['middle_time'] - pd.Timedelta(minutes=30)
    agg_data['max_time'] = agg_data['middle_time'] + pd.Timedelta(minutes=30)
    return agg_data


def plot_bar_chart_meals(x, y, most_occuring):
    colors_bar = ['lightgray'] * 4
    for i in most_occuring:
        colors_bar[i] = '#5c636a'
    layout = go.Layout(width=200, height=100, margin=dict(l=10, r=10, t=25, b=10))
    fig = go.Figure(layout=layout)
    fig.add_trace(
        go.Bar(
            x=x,
            y=y,
            marker_color=colors_bar
            # orientation='h'
        )
    )

    fig.update_layout(
        plot_bgcolor='white',
        yaxis=dict(
            showticklabels=False,
        ),
        font=dict(
            size=8
        )
    )
    return fig


def count_frequency(dataset_cluster):
    meals = []
    # TODO: connect to variable 'times_of_day' in variables
    for item in dataset_cluster:
        hour = item.iloc[0].timestamp.hour + 1
        if 5 <= hour <= 11:
            meal = 'breakfast'
        elif 11 < hour <= 17:
            meal = 'lunch'
        elif 17 < hour <= 22:
            meal = 'dinner'
        else:
            meal = 'night'
        meals.append(meal)
    d = {x: meals.count(x) for x in meals}
    # meal_types = ['night', 'dinner', 'lunch', 'breakfast']
    meal_types = ['breakfast', 'lunch', 'dinner', 'night']
    for key in meal_types:
        if key not in d:
            d[key] = 0
    values = [d[key] for key in meal_types]
    # keys, values = d.keys(), d.values()
    # values = list(values)
    # values_sum = sum(values)
    # values = [item / values_sum for item in values]
    return meal_types, list(values)


def count_treatment_sums(log_list, log_type):
    sums = []
    for logs in log_list:
        sums.append(logs[log_type].sum())
    avg = sum(sums) / len(log_list)
    return avg


def get_time_between_carbs_bolus(carb_list, bolus_list):
    time_between = []
    for carb_df, bolus_df in zip(carb_list, bolus_list):
        time_carbs = carb_df['timestamp'].iloc[0]
        time_bolus = bolus_df['timestamp'].iloc[0]
        time_between.append(time_carbs - time_bolus)
    avg = sum(time_between, timedelta(0))/len(time_between)
    return int(avg.total_seconds() / 60)


def get_start_end_bg(bg_list):
    start_bg = []
    end_bg = []
    for df in bg_list:
        start_bg.append(df['sgv'].iloc[0])
        end_bg.append(df['sgv'].iloc[-1])
    avg_start = round(sum(start_bg) / len(start_bg))
    avg_end = round(sum(end_bg) / len(end_bg))
    return avg_start, avg_end


def get_dataset(logs_):
    indices = {}
    for logs, log_type in zip([logs_sgv, logs_insulin, logs_carbs], ['sgv', 'insulin', 'carbs']):
        indices[log_type] = [get_log_indices(logs, logs_.start_date), get_log_indices(logs, logs_.end_date)]

    dataset_clusters = get_insight_dataset(*indices['sgv'])

    return dataset_clusters, indices


def filter_function_time_of_day(logs, filter):
    logs = logs[logs['time_of_day'].isin(filter)]
    return logs


def filter_function_meal_size(logs, filter):
    logs = logs[logs['carbs', 'sum'].between(filter[0], filter[1])]
    return logs


def get_insight_data_meals(filter_time_of_day=None, filter_meal_size=None):
    logs_meals = get_logs_meals(start_date_insights, end_date, time_before_meal, time_after_meal)
    dataset_unfiltered, _ = get_dataset(logs_meals)

    if filter_meal_size:
        logs_meals = filter_function_meal_size(logs_meals, filter_meal_size)
    if filter_time_of_day:
        logs_meals = filter_function_time_of_day(logs_meals, filter_time_of_day)

    dataset_clusters, indices = get_dataset(logs_meals)

    if not dataset_clusters:
        n_clusters = 0
        return [n_clusters] + [None] * 8
    clusters = get_insight_clusters(dataset_clusters, max_d=200)
    n_clusters_ = len(np.unique(clusters))

    dataset = get_insight_dataset(*indices['sgv'], cluster=False)
    graphs_insights_meals = []
    graphs_meal_overview = []
    carbs_sums = []
    bolus_sums = []
    time_between = []
    start_bgs = []
    end_bgs = []
    most_occurring = []
    logs_indices = get_logs_from_indices(indices)

    for i in range(n_clusters_):
        # dataset_cluster = dataset[clusters == i + 1]
        dataset_cluster = [j for (j, v) in zip(logs_indices['sgv'], clusters == i + 1) if v]
        dataset_cluster_carbs = [j for (j, v) in zip(logs_indices['carbs'], clusters == i + 1) if v]
        dataset_cluster_bolus = [j for (j, v) in zip(logs_indices['insulin'], clusters == i + 1) if v]
        graphs_insights_meals.append(draw_hierarchical_pattern_overview(dataset_cluster, dataset[clusters == i + 1], dataset_cluster_carbs, dataset_cluster_bolus, 'Meal start', time_before_meal))
        x, y = count_frequency(dataset_cluster)
        most_occurring_i = [i for i, x in enumerate(y) if x == max(y)]
        most_occurring.append(x[y.index(max(y))].upper())
        graphs_meal_overview.append(plot_bar_chart_meals(x, y, most_occurring_i))
        carbs_sums.append(count_treatment_sums(dataset_cluster_carbs, 'carbs'))
        bolus_sums.append(count_treatment_sums(dataset_cluster_bolus, 'bolus'))
        time_between.append(get_time_between_carbs_bolus(dataset_cluster_carbs, dataset_cluster_bolus))
        start_bg, end_bg = get_start_end_bg(dataset_cluster)
        start_bgs.append(start_bg)
        end_bgs.append(end_bg)
    graphs_all_curves = get_curve_overview_plot(dataset_clusters, dataset_unfiltered)

    graphs_meal_overview += [{}] * (num_insight_patterns - n_clusters_)
    graphs_insights_meals += [{}] * (num_insight_patterns - n_clusters_)
    start_bgs += [1] * (num_insight_patterns - n_clusters_)
    time_between += [1] * (num_insight_patterns - n_clusters_)
    carbs_sums += [1] * (num_insight_patterns - n_clusters_)
    end_bgs += [1] * (num_insight_patterns - n_clusters_)
    bolus_sums += [1] * (num_insight_patterns - n_clusters_)

    return n_clusters_, graphs_meal_overview, graphs_all_curves, graphs_insights_meals, start_bgs, time_between, carbs_sums, end_bgs, bolus_sums


def get_insight_data_hypos(filter_time_of_day=None):
    logs_hypos, hypo_starts = get_logs_hypos(start_date_insights, end_date, time_before_hypo, time_after_hypo)
    dataset_unfiltered, _ = get_dataset(logs_hypos)

    if filter_time_of_day:
        logs_hypos = filter_function_time_of_day(logs_hypos, filter_time_of_day)
        hypo_starts = logs_hypos['start_date']

    indices = {}
    for logs, log_type in zip([logs_sgv, logs_insulin, logs_carbs], ['sgv', 'insulin', 'carbs']):
        indices[log_type] = [get_log_indices(logs, logs_hypos.start_date), get_log_indices(logs, logs_hypos.end_date)]
    dataset_clusters, _ = get_dataset(logs_hypos)

    if not dataset_clusters:
        n_clusters = 0
        return [n_clusters] + [None] * 8
    clusters = get_insight_clusters(dataset_clusters)
    n_clusters_ = len(np.unique(clusters))

    dataset = get_insight_dataset(*indices['sgv'], cluster=False)
    graphs_insights_meals = []
    graphs_meal_overview = []
    carb_avg_before = []
    carb_avg_after = []
    bolus_avg_before = []
    bolus_avg_after =[]
    time_between = []
    start_bgs = []
    end_bgs = []
    most_occurring = []
    logs_indices = get_logs_from_indices(indices)

    for i in range(n_clusters_):
        hypo_starts_cluster = hypo_starts.iloc[clusters == i + 1].tolist()
        hypo_starts_cluster = [item + timedelta(hours=time_before_hypo) for item in hypo_starts_cluster]
        dataset_cluster = [j for (j, v) in zip(logs_indices['sgv'], clusters == i + 1) if v]
        dataset_cluster_carbs = [j for (j, v) in zip(logs_indices['carbs'], clusters == i + 1) if v]
        dataset_cluster_bolus = [j for (j, v) in zip(logs_indices['insulin'], clusters == i + 1) if v]
        graphs_insights_meals.append(draw_hierarchical_pattern_overview(dataset_cluster, dataset[clusters == i + 1], dataset_cluster_carbs, dataset_cluster_bolus, 'Hypo start', time_before_hypo))
        x, y = count_frequency(dataset_cluster)
        most_occurring_i = [i for i, x in enumerate(y) if x == max(y)]
        most_occurring.append(x[y.index(max(y))].upper())
        graphs_meal_overview.append(plot_bar_chart_meals(x, y, most_occurring_i))
        carb_avg_before_tmp, carb_avg_after_tmp = hypo_event_sums(hypo_starts_cluster, dataset_cluster_carbs, 'carbs')
        bolus_avg_before_tmp, bolus_avg_after_tmp = hypo_event_sums(hypo_starts_cluster, dataset_cluster_bolus, 'bolus')
        carb_avg_before.append(carb_avg_before_tmp)
        carb_avg_after.append(carb_avg_after_tmp)
        bolus_avg_before.append(bolus_avg_before_tmp)
        bolus_avg_after.append(bolus_avg_after_tmp)
        start_bg, end_bg = get_start_end_bg(dataset_cluster)
        start_bgs.append(start_bg)
        end_bgs.append(end_bg)
    graphs_all_curves = get_curve_overview_plot(dataset_clusters, dataset_unfiltered, insights_type='hypos')

    graphs_meal_overview += [{}] * (num_insight_patterns - n_clusters_)
    graphs_insights_meals += [{}] * (num_insight_patterns - n_clusters_)
    start_bgs += [1] * (num_insight_patterns - n_clusters_)
    time_between += [1] * (num_insight_patterns - n_clusters_)
    carb_avg_before += [1] * (num_insight_patterns - n_clusters_)
    carb_avg_after += [1] * (num_insight_patterns - n_clusters_)
    end_bgs += [1] * (num_insight_patterns - n_clusters_)
    bolus_avg_before += [1] * (num_insight_patterns - n_clusters_)
    bolus_avg_after += [1] * (num_insight_patterns - n_clusters_)

    return n_clusters_, graphs_meal_overview, graphs_all_curves, graphs_insights_meals, start_bgs, end_bgs, carb_avg_before, carb_avg_after, bolus_avg_before, bolus_avg_after

def hypo_event_sums(hypo_starts, logs, log_type):
    entries_before = []
    entries_after = []
    for events, hypo_start in zip(logs, hypo_starts):
        if not events.empty:
            for index, row in events.iterrows():
                if events.loc[index].timestamp < hypo_start:
                    entries_before.append(events.loc[index][log_type])
                else:
                    entries_after.append(events.loc[index][log_type])
    avg_before = sum(entries_before) / len(logs)
    avg_after = sum(entries_after) / len(logs)
    return avg_before, avg_after


def get_curve_overview_plot(dataset, dataset_unfiltered, insights_type='meals'):
    layout = go.Layout(width=220, height=200, margin=dict(l=0, r=0, t=25, b=10))
    fig = go.Figure(layout=layout)

    if insights_type == 'meals':
        x = np.linspace(time_before_meal * (-1), len(dataset_unfiltered[0]) * 5 / 60 - time_before_meal, len(dataset_unfiltered[0]))
        tickvals = [i for i in range(int(x[0]), int(x[-1]) + 1)]
        ticklabels = [item if item != 0 else 'Meal start' for item in tickvals]
    else:
        x = np.linspace(time_before_hypo * (-1), len(dataset_unfiltered[0]) * 5 / 60 - time_before_hypo, len(dataset_unfiltered[0]))
        tickvals = [i for i in range(int(x[0]), int(x[-1]) + 1)]
        ticklabels = [item if item != 0 else 'Hypo start' for item in tickvals]

    for data in dataset_unfiltered:
        fig.add_trace(
            go.Scatter(
                x=x,
                y=data,
                hoverinfo='skip',
                mode='lines',
                line=dict(color='lightgray')
            )
        )

    for data in dataset:
        fig.add_trace(
            go.Scatter(
                x=x,
                y=data,
                hoverinfo='skip',
                mode='lines',
                line=dict(color='rgba(92, 99, 106, 0.3)')
            )
        )

    # vertical line
    fig.add_vline(x=0, line_dash="dash")

    fig.update_layout(
        showlegend=False,
        plot_bgcolor=colors['background'],
        paper_bgcolor=colors['background'],
        yaxis=dict(
            tickfont=dict(size=8)
        ),
        xaxis=dict(
            tickvals=tickvals,
            ticktext=ticklabels,
            title='<b>Hours</b>',
            title_font_size=10,
            range=[x[0], x[-1]]
        ),
    )
    return fig


def get_time_of_day_from_number(list_of_numbers):
    time_of_day_dict = {
        1: 'morning',
        2: 'noon',
        3: 'evening',
        4: 'night'
    }
    return [time_of_day_dict[number] for number in list_of_numbers]