import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import interpolate as interp

from colors import colors, colors_agp, colors_agp_bolus
from helpers import get_infos_from_group, get_df_between_dates
from preprocessing import logs_sgv, logs_insulin, logs_carbs, logs_insulin_activity, days, start_date, end_date
from variables import target_range, target_range_extended, font


def calculate_stats(logs, log_type, group, seasonal=True):
    def define_groups(logs):
        logs['hour'] = logs['timestamp'].apply(lambda x: x.hour)
        logs['week'] = logs['timestamp'].apply(lambda x: x.weekday())
        logs['day'] = logs['timestamp'].apply(lambda x: x.strftime("%m/%d/%Y"))
        # logs['week'] = logs['timestamp'].apply(lambda x: int(x.strftime("%V")))
        logs['month'] = logs['timestamp'].apply(lambda x: x.month)
        logs['quarter'] = logs['timestamp'].apply(lambda x: x.quarter)
        return logs

    def std(df):
        return np.sqrt(df.var())

    def percentile(n):
        def percentile_(x):
            return np.percentile(x, n)

        percentile_.__name__ = 'p_%s' % n
        return percentile_

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

    if seasonal:
        logs = define_groups(logs)
        if group == 'hour':
            stats = logs.groupby(group).agg([percentile(10), percentile(25), percentile(50), percentile(75), percentile(90)])
        else:
            if log_type == 'sgv':
                stats = logs.groupby([group]).agg([time_very_low, time_low, time_in_range, time_high, time_very_high])
            else:
                stats = logs.groupby(['day', group]).agg(['sum']).groupby(group).agg(['mean'])
        return stats[log_type]
    else:
        logs = add_placeholder_dates(logs, log_type)
        logs = define_groups(logs)

        dates = pd.DataFrame(days, columns=['timestamp'])
        dates = define_groups(dates)

        if log_type == 'sgv':
            # agg = 'mean'
            stats = logs.groupby([(logs[group].shift() != logs[group]).cumsum(), pd.cut(logs['sgv'], [0, 70, 180, 400])]).count()
            # stats.index = stats.index.droplevel(0)
            stats = stats.sgv
            # list(stats.loc[(slice(None), stats.index[0][1])])
            df_tir = pd.DataFrame({
                'tur': list(stats.loc[(slice(None), stats.index[0][1])]),
                'tir': list(stats.loc[(slice(None), stats.index[1][1])]),
                'tar': list(stats.loc[(slice(None), stats.index[2][1])])
            })
            df_tir["sum"] = df_tir.sum(axis=1)
            df_tir = df_tir.iloc[:, :3].div(df_tir['sum'], axis=0)

            counts = dates.groupby((dates[group].shift() != dates[group]).cumsum()).agg(['count'])[group].reset_index(drop=True).squeeze().tolist()

            tur = [[mean] * count for mean, count in zip(df_tir.tur, counts)]
            tir = [[mean] * count for mean, count in zip(df_tir.tir, counts)]
            tar = [[mean] * count for mean, count in zip(df_tir.tar, counts)]

            return [
                [item for sublist in tur for item in sublist],
                [item for sublist in tir for item in sublist],
                [item for sublist in tar for item in sublist]
            ]

        else:
            agg = 'sum'
            stats = logs.groupby((logs[group].shift() != logs[group]).cumsum()).agg([agg])
            stats.columns = stats.columns.droplevel(1)
            means = stats[[log_type]].reset_index(drop=True).squeeze().tolist()
            counts = dates.groupby((dates[group].shift() != dates[group]).cumsum()).agg(['count'])[group].reset_index(drop=True).squeeze().tolist()
            tmp = [[mean] * count for mean, count in zip(means, counts)]
            # logs['avg'] = [item for sublist in tmp for item in sublist]
            return [item for sublist in tmp for item in sublist]


def add_placeholder_dates(logs, log_type):
    extra_rows = pd.DataFrame({
        'timestamp': days,
        log_type: [np.nan] * len(days)
    })
    logs = pd.concat([logs, extra_rows])
    logs = logs.sort_values(by="timestamp")
    return logs


def get_heatmap_data(logs, log_type, group):
    stats = calculate_stats(logs, log_type, group, seasonal=False)


def calculate_agp(logs, log_type, smoothed=False, h=24, res=10000):
    def smooth(x):
        x_new = np.convolve(x, np.array([1.0, 4.0, 1.0]) / 6.0, 'valid')
        return np.append(np.append(x[0], x_new), x[-1])

    stats = calculate_stats(logs, log_type, 'hour')
    stats = stats.sort_index()
    hours = list(stats.index)

    p90 = smooth(stats.p_90.values) if smoothed else stats.p_90
    p75 = smooth(stats.p_75.values) if smoothed else stats.p_75
    p50 = smooth(stats.p_50.values) if smoothed else stats.p_50
    p25 = smooth(stats.p_25.values) if smoothed else stats.p_25
    p10 = smooth(stats.p_10.values) if smoothed else stats.p_10

    hours = np.append(hours, h)
    p90 = np.append(p90, p90[0])
    p75 = np.append(p75, p75[0])
    p50 = np.append(p50, p50[0])
    p25 = np.append(p25, p25[0])
    p10 = np.append(p10, p10[0])

    interpFun = lambda x, y: interp.CubicSpline(x, y, bc_type='periodic')  # if smoothed
    f90 = interpFun(hours, p90)  # bounds_error=False, fill_value=(p90[0], p90[0]))
    f75 = interpFun(hours, p75)  # , kind='cubic' if smoothed else 'linear', )
    f50 = interpFun(hours, p50)  # , kind='cubic' if smoothed else 'linear', fill_value=np.min(p50))
    f25 = interpFun(hours, p25)  # , kind='cubic' if smoothed else 'linear', fill_value=np.min(p25))
    f10 = interpFun(hours, p10)  # , kind='cubic' if smoothed else 'linear', bounds_error=False,fill_value=(p10[0], p10[0]))

    x = np.linspace(0, h, res)

    p10 = f10(x)
    p25 = f25(x)
    p50 = f50(x)
    p75 = f75(x)
    p90 = f90(x)

    return x, p10, p25, p50, p75, p90


def draw_stacked_tir(group, sgv_logs):
    logs = calculate_stats(sgv_logs, 'sgv', group, seasonal=True).reset_index().rename(columns={'index': group})

    names = ['time_very_low', 'time_low', 'time_in_range', 'time_high', 'time_very_high']
    color_names = ['bg_very_low', 'bg_low', 'bg_target', 'bg_high', 'bg_very_high']
    figure = []
    base = np.zeros(len(logs))
    for i in range(len(names)):
        figure.append(go.Bar(name=names[i], x=logs[group], y=logs[names[i]], offsetgroup=0, base=base, width=0.4, marker_color=colors[color_names[i]]))
        base += np.array(logs[names[i]])
    return figure


def draw_carb_insulin_bars(group, carbs_logs, insulin_logs):
    carb_logs = calculate_stats(carbs_logs, 'carbs', group, seasonal=True).reset_index().rename(columns={'index': group})
    insulin_logs = calculate_stats(insulin_logs, 'bolus', group, seasonal=True).reset_index().rename(columns={'index': group})

    names = ['carbs', 'bolus']
    figures = []
    factor = [1, 10]
    for i, logs in enumerate([carb_logs, insulin_logs]):
        figures.append(go.Bar(name=names[i], x=logs[group], y=logs[('sum', 'mean')] * factor[i], offset=i * 0.2 - 0.2, marker_color=colors[names[i]], width=0.2))
    return figures


def draw_seasonal_graph(group, d_start, d_end):
    sgv_logs = get_df_between_dates(logs_sgv, d_start, d_end)
    carbs_logs = get_df_between_dates(logs_carbs, d_start, d_end)
    insulin_logs = get_df_between_dates(logs_insulin, d_start, d_end)
    fig = make_subplots(rows=2, cols=1, shared_xaxes=False, row_heights=[0.4, 0.6], vertical_spacing=0.01)
    x_values, x_labels = get_infos_from_group(group)
    fig.update_layout(
        showlegend=False,
        width=330,
        height=250,
        xaxis1=dict(
            showticklabels=False,
            range=[x_values[0] - 0.5, x_values[-1] + 0.5],
            fixedrange=True
        ),
        xaxis2=dict(
            # tickmode='array',
            tickvals=x_values,
            ticktext=x_labels,
            range=[x_values[0] - 0.5, x_values[-1] + 0.5],
            fixedrange=True
        ),
        yaxis=dict(
            showticklabels=False,
            range=[0, 1],
            fixedrange=True
        ),
        yaxis2=dict(
            showticklabels=False,
            range=[0, 300],
            fixedrange=True
        ),
        margin=dict(t=15, b=10, l=15, r=10),
        plot_bgcolor=colors['background'],
        paper_bgcolor=colors['background'],
    )

    fig_stacked_tir = draw_stacked_tir(group, sgv_logs)
    for f in fig_stacked_tir:
        fig.add_trace(f, row=1, col=1)

    fig_carbs_insulin = draw_carb_insulin_bars(group, carbs_logs, insulin_logs)
    for f in fig_carbs_insulin:
        fig.add_trace(f, row=2, col=1)

    return fig


def draw_agp_carbs_bolus():
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, subplot_titles=('<b>Carbs</b> g', '<b>Bolus</b> U'))
    fig = draw_barcode_plot(fig, get_df_between_dates(logs_carbs, start_date, end_date), 'carbs', row=1)
    fig = draw_barcode_plot(fig, get_df_between_dates(logs_insulin, start_date, end_date), 'bolus', row=2)

    fig.update_layout(
        showlegend=False,
        width=620,
        height=150,
        yaxis=dict(
            showticklabels=False,
            range=[1, 2],
            fixedrange=True
            # title='Bolus'
        ),
        yaxis2=dict(
            showticklabels=False,
            range=[1, 2],
            fixedrange=True
            # title='Carbs'
        ),
        margin=dict(t=15, b=10, l=0, r=0),
    )

    fig.layout.annotations[0].update(x=0.03, font=dict(family=font, size=10))
    fig.layout.annotations[1].update(x=0.03, font=dict(family=font, size=10))
    return fig


def draw_seasonal_graph_day(start_date, end_date, weekday_filter=None):
    fig_agp = make_subplots(rows=3, cols=1, shared_xaxes=True, row_heights=[0.8, 0.1, 0.1], vertical_spacing=0.01)
    fig_agp = draw_agp(fig_agp, calculate_agp(get_df_between_dates(logs_sgv, start_date, end_date, weekday_filter), 'sgv'), row=1, hover=False)
    logs_insulin_activity_extended = logs_insulin_activity.copy()
    logs_insulin_activity_extended['insulin_activity'] = logs_insulin_activity_extended['insulin_activity'] * 100
    fig_agp = draw_barcode_plot(fig_agp, get_df_between_dates(logs_carbs, start_date, end_date, weekday_filter), 'carbs', row=2)
    fig_agp = draw_barcode_plot(fig_agp, get_df_between_dates(logs_insulin, start_date, end_date, weekday_filter), 'bolus', row=3)
    x_values, x_labels = get_infos_from_group('day')
    fig_agp.update_layout(
        yaxis_range=[30, 300],
        showlegend=False,
        width=575,
        height=140,
        xaxis3=dict(
            # tickmode='array',
            tickvals=x_values,
            showgrid=True,
            range=[0, 24],
            visible=False
        ),
        yaxis3=dict(
            showticklabels=False,
            range=[1, 2],
            fixedrange=True,
            visible=False
        ),
        yaxis2=dict(
            showticklabels=False,
            range=[1, 2],
            fixedrange=True,
            visible=False
        ),
        yaxis=dict(
            tickfont_size=8,
            tickvals=[0, 70, 180, 300],
            fixedrange=True,
            visible=False
        ),
        font=dict(
            family=font,
        ),
        margin=dict(t=5, b=5, l=0, r=0),
        plot_bgcolor=colors['background'],
    )
    fig_agp.update_layout(dragmode="select", clickmode='event+select')
    return fig_agp


def draw_full_agp(start_date, end_date, weekday_filter=None):
    fig_agp = make_subplots(rows=4, cols=1, shared_xaxes=True, row_heights=[0.5, 0.1, 0.3, 0.1], vertical_spacing=0.07, subplot_titles=('<b>Glucose</b> mg/dL', '<b>Carbs</b> g', '<b>Bolus</b> U', ''))
    fig_agp = draw_agp(fig_agp, calculate_agp(get_df_between_dates(logs_sgv, start_date, end_date, weekday_filter), 'sgv'), row=1)
    logs_insulin_activity_extended = logs_insulin_activity.copy()
    logs_insulin_activity_extended['insulin_activity'] = logs_insulin_activity_extended['insulin_activity'] * 100
    fig_agp = draw_agp(fig_agp, calculate_agp(logs_insulin_activity_extended, 'insulin_activity'), colors=colors_agp_bolus, row=3)

    fig_agp = draw_barcode_plot(fig_agp, get_df_between_dates(logs_carbs, start_date, end_date, weekday_filter), 'carbs', row=2, spacing=0.08)
    fig_agp = draw_barcode_plot(fig_agp, get_df_between_dates(logs_insulin, start_date, end_date, weekday_filter), 'bolus', row=4, spacing=0.08)
    x_values = [0, 3, 6, 9, 12, 15, 18, 21, 24]
    x_labels = ['0:00', '3:00', '6:00', '9:00', '12:00', '15:00', '18:00', '21:00', '24:00']
    # x_labels = ['0 am', '3 am', '6 am', '9 am', '12 am', '3 pm', '6 pm', '9 pm', '0 pm']
    fig_agp.update_layout(
        yaxis_range=[30, 300],
        showlegend=False,
        # width=620,
        # height=400,
        xaxis4=dict(
            tickmode='array',
            tickvals=x_values,
            ticktext=x_labels,
            showgrid=True,
            range=[0, 24],
            # visible=False
        ),
        yaxis4=dict(
            showticklabels=False,
            range=[1, 2],
            fixedrange=True,
            visible=False
        ),
        yaxis2=dict(
            showticklabels=False,
            range=[1, 2],
            fixedrange=True,
            visible=False
        ),
        yaxis3=dict(
            range=[0, 10],
            showticklabels=False,
            fixedrange=True,
            visible=False
        ),
        yaxis=dict(
            tickfont_size=8,
            tickvals=[0, 70, 180, 300],
            fixedrange=True
            # title='Glucose'
        ),
        font=dict(
            family=font,
        ),
        margin=dict(t=30, b=5, l=0, r=10),
        plot_bgcolor=colors['background'],
    )
    # fig_agp.update_layout(dragmode="select", clickmode='event+select')
    fig_agp.layout.annotations[0].update(x=0.06, font=dict(family=font, size=10))
    fig_agp.layout.annotations[1].update(x=0.03, font=dict(family=font, size=10))
    fig_agp.layout.annotations[2].update(x=0.03, font=dict(family=font, size=10))
    return fig_agp


def agp_xaxis(width=590, margin_left=10):
    fig_agp = make_subplots(rows=1, cols=1)
    fig_agp = draw_agp(fig_agp, calculate_agp(get_df_between_dates(logs_sgv, start_date, end_date), 'sgv'), row=1)

    x_values = [0.5, 3, 6, 9, 12, 15, 18, 21, 24]
    x_labels = ['0:00', '3:00', '6:00', '9:00', '12:00', '15:00', '18:00', '21:00', '24:00']
    fig_agp.update_layout(
        showlegend=False,
        width=width,
        height=30,
        xaxis=dict(
            tickvals=x_values,
            showgrid=True,
            ticktext=x_labels,
            range=[0, 24],
            visible=True
        ),
        yaxis=dict(
            visible=False,
        ),
        font=dict(
            family=font,
        ),
        margin=dict(t=30, b=30, l=margin_left, r=0),
        plot_bgcolor=colors['background'],
    )
    fig_agp.update_layout(dragmode="select", clickmode='event+select')
    return fig_agp


def get_barcode_plot_data(logs, log_type):
    a = logs.timestamp
    x = a.dt.hour + a.dt.minute / 60.0
    x = x.to_numpy()

    alphas = logs[log_type].to_numpy()
    alphas = alphas / alphas.max() * 0.8
    return x, alphas


def draw_barcode_plot(fig, logs, log_type, row, spacing=0.25):
    x, alphas = get_barcode_plot_data(logs, log_type)

    for i in np.arange(1, 2, spacing):
        fig.add_trace(
            go.Scattergl(
                x=x,
                y=[i] * len(x),
                hoverinfo='skip',
                mode='markers',
                marker=dict(size=6, symbol='square-dot', opacity=alphas, color=colors[log_type]),
            ),
            row=row, col=1,
        )
    # fig.update_layout(plot_bgcolor=colors['background'])
    return fig


def update_seasonal_graph_daily(data, start_date, end_date):
    sgv_logs = get_df_between_dates(logs_sgv, start_date, end_date)
    agp_data = get_agp_plot_data(calculate_agp(sgv_logs, 'sgv'))

    for i, d in enumerate(agp_data):
        data[i].update({
            'x': d[0],
            'y': d[1],
        })

    carb_logs = get_df_between_dates(logs_carbs, start_date, end_date)
    x, alphas = get_barcode_plot_data(carb_logs, 'carbs')

    num_traces = len(np.arange(1, 2, 0.12))

    for i in range(7, num_traces + 7):
        data[i].update({
            'x': x,
        })
        data[i]['marker'].update({'opacity': alphas})

    insulin_logs = get_df_between_dates(logs_insulin, start_date, end_date)
    x, alphas = get_barcode_plot_data(insulin_logs, 'bolus')

    for i in range(7 + num_traces, 7 + 2 * num_traces):
        data[i].update({
            'x': x,
        })
        data[i]['marker'].update({'opacity': alphas})
    return data


def get_agp_plot_data(data):
    x, p10, p25, p50, p75, p90 = data

    x_long = np.append(x, np.flip(x))

    y1 = [min(target_range[1], y) for y in p90]
    y1 = np.array([max(target_range[0], y) for y in y1])

    y2 = [max(target_range[0], y) for y in p10]
    y2 = np.array([min(target_range[1], y) for y in y2])

    y3 = [min(target_range[1], y) for y in p75]
    y3 = np.array([max(target_range[0], y) for y in y3])

    y4 = [max(target_range[0], y) for y in p25]
    y4 = np.array([min(target_range[1], y) for y in y4])

    y5 = np.array([max(target_range[1], y) for y in p25])
    y6 = np.array([max(target_range[1], y) for y in p75])

    y7 = np.array([min(target_range[0], y) for y in p25])
    y8 = np.array([min(target_range[0], y) for y in p75])

    y9 = np.array([max(target_range[1], y) for y in p10])
    y10 = np.array([max(target_range[1], y) for y in p90])

    y11 = np.array([min(target_range[0], y) for y in p10])
    y12 = np.array([min(target_range[0], y) for y in p90])

    return [
        [x_long, np.append(y1, np.flip(y2))],
        [x_long, np.append(y3, np.flip(y4))],
        [x, p50],
        [x_long, np.append(y5, np.flip(y6))],
        [x_long, np.append(y7, np.flip(y8))],
        [x_long, np.append(y9, np.flip(y10))],
        [x_long, np.append(y11, np.flip(y12))],
    ]


def draw_agp(fig_agp, data, colors=colors_agp, row=False, hover=False):
    d1, d2, d3, d4, d5, d6, d7 = get_agp_plot_data(data)

    trace_1 = go.Scatter(
        x=d1[0],
        y=d1[1],
        fillcolor=colors['in_range_90th'],
        fill='toself',
        hoverinfo='skip',
        line=dict(color='rgba(255,255,255, 0)'),
    )

    trace_2 = go.Scatter(
        x=d2[0],
        y=d2[1],
        fillcolor=colors['in_range_75th'],
        fill='toself',
        hoverinfo='skip',
        line=dict(color='rgba(255,255,255, 0)'),
    )

    trace_3 = go.Scatter(
        x=d3[0],
        y=d3[1],
        mode='lines',
        hoverinfo='skip',
        line=dict(color=colors['in_range_median']),
    )

    trace_4 = go.Scatter(
        x=d4[0],
        y=d4[1],
        mode="lines", line=dict(color='rgba(255, 255, 255, 0.5)'), hoverinfo='skip',
        fillcolor=colors['above_range_75th'], fill='toself',
        showlegend=False
    )

    trace_5 = go.Scatter(
        x=d5[0],
        y=d5[1],
        mode="lines", line=dict(color='rgba(255, 255, 255, 0.5)'), hoverinfo='skip',
        fillcolor=colors['under_range_75th'], fill='toself',
        showlegend=False
    )

    trace_6 = go.Scatter(
        x=d6[0],
        y=d6[1],
        mode="lines", line=dict(color='rgba(255, 255, 255, 0.5)'), hoverinfo='skip',
        fillcolor=colors['above_range_90th'], fill='toself',
        showlegend=False
    )

    trace_7 = go.Scatter(
        x=d7[0],
        y=d7[1],
        mode="lines", line=dict(color='rgba(255, 255, 255, 0.5)'), hoverinfo='skip',
        fillcolor=colors['under_range_90th'], fill='toself',
        showlegend=False
    )

    if row:
        fig_agp.add_trace(trace_1, row=row, col=1)
        fig_agp.add_trace(trace_2, row=row, col=1)
        fig_agp.add_trace(trace_3, row=row, col=1)
        fig_agp.add_trace(trace_4, row=row, col=1)
        fig_agp.add_trace(trace_5, row=row, col=1)
        fig_agp.add_trace(trace_6, row=row, col=1)
        fig_agp.add_trace(trace_7, row=row, col=1)
    else:
        fig_agp.add_trace(trace_1)
        fig_agp.add_trace(trace_2)
        fig_agp.add_trace(trace_3)
        fig_agp.add_trace(trace_4)
        fig_agp.add_trace(trace_5)
        fig_agp.add_trace(trace_6)
        fig_agp.add_trace(trace_7)

    if hover:
        fig_agp.add_trace(
            go.Scatter(
                x=[0, 24, 24, 0],
                y=[0, 0, 400, 400],
                hoveron='fills',
                text=' ',
                hovertemplate="",
                name='Click and drag to select a region of interest.<br />Double click to go back to overview.',
                mode="lines", line=dict(color='rgba(255, 255, 255, 0)'),
                fillcolor='rgba(0,0,0,0)',
                fill='toself',
                showlegend=False
            )
        )
        fig_agp.update_layout(
            hoverlabel=dict(
                bgcolor='rgba(255, 255, 255, 0.2)',
                bordercolor='rgba(255, 255, 255, 0)',
                namelength=-1,
            ),
        )

    return fig_agp


def draw_boxplot(group):
    fig = px.box(logs_sgv, x=group, y='sgv', points=False)
    fig.update_traces(quartilemethod="inclusive")
    return fig
