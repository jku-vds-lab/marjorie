import numpy as np
import pandas as pd
from datetime import date, timedelta, datetime
import datetime as dt

from plotly.subplots import make_subplots
from sklearn import decomposition
from sklearn import manifold
from sklearn.cluster import KMeans, AffinityPropagation, MeanShift, SpectralClustering, AgglomerativeClustering, DBSCAN, OPTICS, Birch, KMeans
import umap

from colors import colors_agp
from helpers import normalize
from preprocessing import logs_sgv, sgv_delta, insulin_activity, insulin_activity_delta, logs_carbs, logs_insulin
from variables import target_range, num_motifs, window_size, num_segments
from aggregations import calculate_agp, draw_agp, get_agp_plot_data
from helpers import convert_datestrings_df, get_deltas
import plotly.graph_objects as go
import plotly.express as px

class SlidingWindow():

    def __init__(self, data, window_size=3, start_time=0, stride=1):
        """
        Sliding window approach: Multiple vectors of length "window_size" are created from "time_data" vector, each shifted by "stride" time steps
        :param time_data: list of time data
        :param window_size: width of sliding window
        :param start_time: index at which the sliding window should start
        :param stride: time steps over which the window should slide
        """
        self.data = np.array(data)
        self.window_size = window_size
        self.start_time = start_time
        self.stride = stride

    def normalize(self, time_data):
        time_data = (time_data - np.min(self.data)) / (np.max(self.data) - np.min(self.data))
        return time_data

    def get_window_matrix(self, time_data):
        sub_windows = self.start_time + np.expand_dims(np.arange(self.window_size), 0) + np.expand_dims(
            np.arange(len(time_data) - self.window_size + 1 - self.start_time), 0).T
        window_matrix = time_data[sub_windows[::self.stride]]
        return window_matrix


class Preprocessing():
    def __init__(self, sgv):
        self.sgv = sgv

    def smoothing(self):
        sgv_pd = pd.DataFrame(self.sgv, columns=['sgv'])
        # sgv_pd['mov_avg'] = sgv_pd['sgv'].rolling(6).sum()
        sgv_pd['mov_avg'] = sgv_pd['sgv'].rolling(6).mean()
        return sgv_pd['mov_avg'].dropna().to_numpy()

class DimensionalityReduction():

    def __init__(self, input_matrix):
        self.input_matrix = input_matrix

    def pca(self):
        pca = decomposition.PCA(n_components=2)
        pos = pca.fit(self.input_matrix).transform(self.input_matrix)
        return pos

    def tsne(self):
        tsne = manifold.TSNE(n_components=2)
        pos = tsne.fit_transform(self.input_matrix)
        return pos

    def u_map(self):
        reducer = umap.UMAP()
        pos = reducer.fit_transform(self.input_matrix)
        return pos


# windows
def create_df_windows():
    num_ids = int(np.floor(len(logs_sgv) / window_size))
    id = np.repeat(range(num_ids), window_size)
    rest = np.repeat(num_ids, len(logs_sgv) - len(id))
    id = np.concatenate((id, rest), axis=0)

    df_windows = logs_sgv.copy()
    df_windows['insulin_activity'] = insulin_activity
    df_windows['carbs'] = 0
    df_windows['bolus'] = 0
    df_windows['id'] = id

    # add carbs
    for date_carbs, val_carbs in zip(logs_carbs.timestamp.to_numpy(), logs_carbs.carbs.to_numpy()):
        diff = df_windows.timestamp.to_numpy() - date_carbs
        idx = np.argmin(abs(diff))
        df_windows.at[df_windows.index[idx], 'carbs'] += val_carbs

    # add bolus
    for date_bolus, val_bolus in zip(logs_insulin.timestamp.to_numpy(), logs_insulin.bolus.to_numpy()):
        diff = df_windows.timestamp.to_numpy() - date_bolus
        idx = np.argmin(abs(diff))
        df_windows.at[df_windows.index[idx], 'bolus'] += val_bolus

    df_windows['insulin_activity_delta'] = get_deltas(df_windows.insulin_activity)
    df_windows['transformed_sgv'] = transform_bg_scale(df_windows.sgv)
    df_windows['sgv_delta'] = get_deltas(df_windows.transformed_sgv)
    return df_windows


def transform_bg_scale(sgv):
    return 1.509 * (np.log(sgv) ** 1.084 - 5.381)


def tur(sgv):
    count_under_range = sgv[sgv < -0.88].count()
    return count_under_range / len(sgv)


def start_end_delta(sgv):
    return sgv.iloc[-1] - sgv.iloc[0]


def lbgi(sgv):
    '''sgv needs to be already transformed'''
    sgv[sgv > 0] = 0
    r = 22.77 * sgv ** 2
    return r.sum() / len(r)


def hbgi(sgv):
    '''sgv needs to be already transformed'''
    sgv[sgv < 0] = 0
    r = 22.77 * sgv ** 2
    return r.sum() / len(r)


most_common = lambda x: x.value_counts().index[0]
n_insulin = lambda x: x.nunique() - 1


def get_features(df_windows):
    stats = df_windows.groupby('id').agg(['mean', 'sum', 'max', 'min', 'std', n_insulin, tur, start_end_delta, lbgi, hbgi])

    features = stats.transformed_sgv[['mean', 'max', 'min', 'std', 'tur', 'start_end_delta']].copy()
    features.columns = ['sgv_' + item for item in features.columns]

    # columns = ['sum', 'max', 'min', 'std']
    # column_names = ['insulin_activity_' + item for item in columns]
    # features[column_names] = stats.insulin_activity[columns].copy()

    columns = ['mean', 'std']
    column_names = ['sgv_delta_' + item for item in columns]
    features[column_names] = stats.sgv_delta[columns].copy()

    return features

def get_fetures_sliding_windows():
    sgv = logs_sgv['sgv'].to_numpy()[-200:]
    window = SlidingWindow(sgv, window_size=window_size)
    sliding_matrix = window.get_window_matrix(sgv)
    matrix_dates = window.get_window_matrix(logs_sgv['timestamp'].to_numpy())[-1000:]



    reduced_matrix = DimensionalityReduction(sliding_matrix).pca()
    clustering = KMeans(n_clusters=num_motifs).fit(reduced_matrix)
    labels = clustering.labels_
    print(clustering)
    return reduced_matrix, clustering


def get_clusters():
    df_windows = create_df_windows()
    features = get_features(df_windows)
    reduced_matrix = DimensionalityReduction(features[:10000]).pca()
    clustering = AgglomerativeClustering(n_clusters=num_motifs).fit(reduced_matrix)

    # (distance_threshold=0, n_clusters=None)
    df_windows['cluster'] = [clustering.labels_[idx] for idx in df_windows.id]
    df_cluster_ids = df_windows.set_index(['cluster', 'id'], inplace=False)
    return df_cluster_ids


def check_filter(filter, segment):
    is_filtered = False
    if filter:
        for key in filter.keys():
            if key == 'time_range':
                end_time = segment.timestamp.iloc[-1].hour + segment.timestamp.iloc[-1].minute / 60
                start_time = segment.timestamp.iloc[0].hour + segment.timestamp.iloc[0].minute / 60
                start_time_lies_in_interval = filter['time_range'][0] <= start_time <= filter['time_range'][1]
                end_time_lies_in_interval = filter['time_range'][0] <= end_time <= filter['time_range'][1]
                if not (start_time_lies_in_interval or end_time_lies_in_interval):
                    is_filtered = True
                # else:
                #     print(start_time, end_time, [filter['time_range'][0], filter['time_range'][1]])
    return is_filtered


def draw_motifs_filtered(df_cluster_ids, filter):
    figures_summaries = []
    filter_idx = []
    segment_highlight_ranges = []
    x_highlight_data = []
    carbs_aggr = []
    insulin_aggr = []

    for cl in range(num_motifs):
        # print('###### motif #####')
        cluster_df = df_cluster_ids[df_cluster_ids.index.get_level_values('cluster') == cl]
        pattern_ids = cluster_df.index.get_level_values('id').value_counts().index.sort_values()

        sgvs = []
        idx = []
        highlight_range = []
        start_dates = []
        carbs = []
        insulin = []

        # print('segment start')
        i = 0
        for pattern_id in pattern_ids:
            segment = cluster_df.loc[cl, pattern_id]
            time_delta = segment.timestamp.iloc[-1] - segment.timestamp.iloc[0]
            contains_data_gap = (time_delta.seconds / 60) > (5 * window_size * 1.3)
            is_filtered = check_filter(filter, segment)

            if not contains_data_gap:
                if not is_filtered:
                    idx.append(i)  # TODO: smth wrong here

                    sgvs.extend(segment.sgv.to_list())
                    highlight_range.append([segment.timestamp.iloc[0], segment.timestamp.iloc[-1]])
                    start_dates.append(segment.timestamp.iloc[0])
                    carbs.append(segment.carbs.sum())
                    insulin.append(segment.bolus.sum())
                i += 1

        carbs_mean = np.array(carbs).mean()
        insulin_mean = np.array(insulin).mean()

        if sgvs:
            fig_sum = update_motif_summary_data(sgvs, len(carbs))
        else:
            fig_sum = None

        figures_summaries.append(fig_sum) # append motif summary

        filter_idx.append(idx)  # append list of segment graphs for this motif
        segment_highlight_ranges.append(highlight_range)  # ranges to highlight in timeline graph
        x_highlight_data.append(start_dates)
        carbs_aggr.append(carbs_mean)
        insulin_aggr.append(insulin_mean)

    return figures_summaries, filter_idx, segment_highlight_ranges, x_highlight_data, carbs_aggr, insulin_aggr

def draw_motifs(df_cluster_ids):
    figures_summaries = []
    figures_segments = []
    segment_highlight_ranges = []
    x_highlight_data = []
    carbs_aggr = []
    insulin_aggr = []

    for cl in range(num_motifs):
        # print('###### motif #####')
        cluster_df = df_cluster_ids[df_cluster_ids.index.get_level_values('cluster') == cl]
        pattern_ids = cluster_df.index.get_level_values('id').value_counts().index

        sgvs = []
        fig_seg = []
        highlight_range = []
        start_dates = []
        carbs = []
        insulin = []

        # print('segment start')
        for i, pattern_id in enumerate(pattern_ids):
            segment = cluster_df.loc[cl, pattern_id]
            time_delta = segment.timestamp.iloc[-1] - segment.timestamp.iloc[0]
            contains_data_gap = (time_delta.seconds / 60) > (5 * window_size * 1.3)

            if not contains_data_gap:
                    fig_seg.append(draw_motif_segment(segment))  # plot each segment in a figure
                    sgvs.extend(segment.sgv.to_list())
                    highlight_range.append([segment.timestamp.iloc[0], segment.timestamp.iloc[-1]])
                    start_dates.append(segment.timestamp.iloc[0])
                    carbs.append(segment.carbs.sum())
                    insulin.append(segment.bolus.sum())

        # order = np.argsort(start_dates)
        # fig_seg = [fig_seg[i] for i in order]
        # highlight_range = [highlight_range[i] for i in order]
        # start_dates = [start_dates[i] for i in order]

        carbs_mean = np.array(carbs).mean()
        insulin_mean = np.array(insulin).mean()

        fig_sum = draw_motif_summary(sgvs, len(carbs))
        figures_summaries.append(fig_sum) # append motif summary

        figures_segments.append(fig_seg)  # append list of segment graphs for this motif
        segment_highlight_ranges.append(highlight_range)  # ranges to highlight in timeline graph
        x_highlight_data.append(start_dates)
        carbs_aggr.append(carbs_mean)
        insulin_aggr.append(insulin_mean)

    return figures_summaries, figures_segments, segment_highlight_ranges, x_highlight_data, carbs_aggr, insulin_aggr


def update_motif_summary_data(sgvs, num_segments):
    dates = [datetime(2000, 1, 1) + timedelta(minutes=x) for x in range(0, 5 * window_size, 5)]
    df = pd.DataFrame(dict(timestamp=(dates * num_segments)[:len(sgvs)], sgv=sgvs))
    agp_stats = calculate_agp(df, 'sgv', h=3, res=100)
    d1, d2, d3, d4, d5, d6, d7 = get_agp_plot_data(agp_stats)

    # for i, d in enumerate(agp_data):
    #     data[i].update([('x', d[0]), ('y', d[1])])

    data = [
        dict(
            x=d1[0],
            y=d1[1],
            fillcolor=colors_agp['in_range_90th'],
            fill='toself',
            hoverinfo='skip',
            line=dict(color='rgba(255,255,255, 0)'),
        ),
        dict(
            x=d2[0],
            y=d2[1],
            fillcolor=colors_agp['in_range_75th'],
            fill='toself',
            hoverinfo='skip',
            line=dict(color='rgba(255,255,255, 0)'),
        ),
        dict(
            x=d3[0],
            y=d3[1],
            mode='lines',
            hoverinfo='skip',
            line=dict(color=colors_agp['in_range_median']),
        ),
        dict(
            x=d4[0],
            y=d4[1],
            mode="lines", line=dict(color='rgba(255, 255, 255, 0.5)'), hoverinfo='skip',
            fillcolor=colors_agp['above_range_75th'], fill='toself',
            showlegend=False
        ),
        dict(
            x=d5[0],
            y=d5[1],
            mode="lines", line=dict(color='rgba(255, 255, 255, 0.5)'), hoverinfo='skip',
            fillcolor=colors_agp['under_range_75th'], fill='toself',
            showlegend=False
        ),
        dict(
            x=d6[0],
            y=d6[1],
            mode="lines", line=dict(color='rgba(255, 255, 255, 0.5)'), hoverinfo='skip',
            fillcolor=colors_agp['above_range_90th'], fill='toself',
            showlegend=False
        ),
        dict(
            x=d7[0],
            y=d7[1],
            mode="lines", line=dict(color='rgba(255, 255, 255, 0.5)'), hoverinfo='skip',
            fillcolor=colors_agp['under_range_90th'], fill='toself',
            showlegend=False
        )
    ]
    return data


def draw_motif_summary(sgvs, num_segments):
    dates = [datetime(2000, 1, 1) + timedelta(minutes=x) for x in range(0, 5 * window_size, 5)]
    df = pd.DataFrame(dict(timestamp=(dates * num_segments)[:len(sgvs)], sgv=sgvs))

    layout = go.Layout(
        showlegend=False,
        width=230, height=150,
        xaxis=go.layout.XAxis(
            showticklabels=False,
            mirror=True,
            fixedrange=True,
            gridwidth=2,
            # ticks='outside',
            showline=True,
        ),
        yaxis=go.layout.YAxis(
            showticklabels=False,
            mirror=True,
            # ticks='outside',
            fixedrange=True,
            tickvals=target_range,
            gridwidth=2,
            showline=True,
        ),
        # dict(l=40, r=20, t=25, b=10),
        margin=dict(t=25, b=10, l=30, r=0),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
    )

    fig_summ = draw_agp(go.Figure(layout=layout, layout_yaxis_range=[0, 350]), calculate_agp(df, 'sgv', h=3, res=100))
    fig_summ.add_vrect(x0=dates[0], x1=dates[1], line_color='white', line_width=20, opacity=1)

    return fig_summ


def update_motif_segment_data(df):
    data = [
        {
            'x': df.timestamp,
            'y': df.sgv,
            'mode': 'lines',
        },
    ]

    carb_df = df.loc[df.carbs != 0][['timestamp', 'carbs']]
    if len(carb_df) > 0:
        data.append(
            {
                'x': carb_df.timestamp,
                'y': carb_df.carbs,
                'marker': dict(opacity=0.8, color='rgba(255,213,114, 1)'),
                'yaxis': 'y2',
                'width': 0.5 * 10 * 3600 * 24,
            },
        )

    insulin_df = df.loc[df.bolus != 0][['timestamp', 'bolus']]
    if len(insulin_df) > 0:
        data.append(
            {
                'x': insulin_df.timestamp,
                'y': insulin_df.bolus,
                'marker': dict(opacity=0.8, color='rgba(90,209,245, 1)'),
                'yaxis': 'y2',
                'width': 0.5 * 10 * 3600 * 24,
            },
        )
    return data


def draw_motif_segment(df):
    layout = go.Layout(width=350, height=150, margin=dict(l=10, r=10, t=25, b=10))
    fig = go.Figure(layout=layout, layout_yaxis_range=[0, 350])
    fig.add_trace(go.Scatter(
        x=df.timestamp,
        y=df.sgv,
        mode='lines',
        connectgaps=False
    ))

    carb_df = df.loc[df.carbs != 0][['timestamp', 'carbs']]
    if len(carb_df) > 0:
        fig.add_trace(
            go.Bar(
                x=carb_df.timestamp,
                y=carb_df.carbs,
                marker=dict(opacity=0.8, color='rgba(255,213,114, 1)'),
                # line=dict(color='rgba(255,213,114, 1)'),
                # xaxis='x2',
                yaxis='y2',
                width=0.5 * 10 * 3600 * 24,
            ),
            # row=1, col=1
        )

        fig.layout['yaxis2'] = {'visible': False, 'side': 'right', 'overlaying': 'y', 'range': [0, logs_carbs.carbs.max() * 2]}

    insulin_df = df.loc[df.bolus != 0][['timestamp', 'bolus']]
    if len(insulin_df) > 0:
        fig.add_trace(
            go.Bar(
                x=insulin_df.timestamp,
                y=insulin_df.bolus,
                marker=dict(opacity=0.8, color='rgba(90,209,245, 1)'),
                # line=dict(color='rgba(90,209,245, 1)'),
                # xaxis='x2',
                yaxis='y3',
                width=0.5 * 10 * 3600 * 24,
            ),
            # row=1, col=1
        )

        fig.layout['yaxis3'] = {'visible': False, 'side': 'right', 'overlaying': 'y', 'range': [0, logs_insulin.bolus.max() * 2]}
    fig.update_layout(showlegend=False,
                      paper_bgcolor='rgba(0,0,0,0)',
                      plot_bgcolor='rgba(0,0,0,0)',
                      margin=dict(t=0, b=50, l=50, r=30),
                      )
    return fig

# def draw_motif_occurences(motif_nr):
#     figures = []
#     for cl in range(3):
#         fig = draw_clustering()
#         figures.append(fig)
#     return figures

# x, p10, p25, p50, p75, p90 = draw_motif_summary()


def draw_clustering(reduced_matrix, clustering):
    layout = go.Layout(showlegend=False, width=620, height=400, margin=dict(l=10, r=10, t=25, b=10))
    fig = go.Figure(layout=layout)
    fig.add_trace(go.Scatter(
        x=reduced_matrix[:, 0],
        y=reduced_matrix[:, 1],
        mode='lines',
        # marker=dict(
        #     size=4,
        #     color=clustering.labels_,
        #     colorscale='Viridis',
        #     showscale=False
        # )
    ))
    return fig
