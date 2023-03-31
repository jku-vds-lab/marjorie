from datetime import timedelta, datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from colors import colors
from helpers import dt_details_function, highlight_data, get_df_of_date
from preprocessing import logs_sgv, logs_insulin, logs_carbs, date_min, date_max, days, colorscales, colorscale, targets, colorscale_categorical, logs_br_default, logs_br
from variables import target_range
import scipy
from scipy.cluster.hierarchy import ward, fcluster

font = 'Verdana, sans-serif'
start_date = date_max - timedelta(days=1)
end_date = date_max

y_range = [-400, 450]

layout = go.Layout(width=760, height=500, margin=dict(t=25, b=10, l=0, r=0), plot_bgcolor='rgba(248,249,250,1)', )


def draw_daily_plot(day, zoom_data=None, cutoff_value=1.5, bar_width_factor=15):
    midnight = datetime.combine(day, datetime.min.time())
    x_range_original = [midnight, midnight + timedelta(days=1)]

    if zoom_data:
        x_range = zoom_data
    else:
        x_range = x_range_original.copy()

    fig_timeline = make_subplots(rows=4, cols=1, shared_xaxes=True, row_heights=[0.4, 0.2, 0.2, 0.2], vertical_spacing=0.05,
                                 subplot_titles=('<b>Glucose</b> mg/dL', '<b>Carbs</b> g', '<b>Bolus</b> U', '<b>Basal</b> U/h'))
    sgv_today = get_df_of_date(logs_sgv, day)
    carbs_today = get_df_of_date(logs_carbs, day)
    insulin_today = get_df_of_date(logs_insulin, day)
    if logs_br is not None:
        br_today = get_df_of_date(logs_br, day)
    br_default_today = get_df_of_date(logs_br_default, day)

    # target range
    fig_timeline.add_trace(go.Scattergl(
        x=np.concatenate([x_range_original, x_range_original[::-1]]),
        y=np.concatenate([[target_range[1], target_range[1]], [target_range[0], target_range[0]][::-1]]),
        line=dict(color='rgba(255,255,255, 1)', dash='dash'),
    ),
        row=1, col=1,
    )

    # sgv
    y_sgv = sgv_today.sgv.fillna(0)
    fig_timeline.add_trace(
        go.Scattergl(
            x=sgv_today.timestamp,
            y=y_sgv,
            # color=logs_sgv.sgv,
            mode='markers',
            marker=dict(size=4, color=sgv_today.sgv, colorscale=colorscale),
            connectgaps=False,
            # xaxis='x2',
            hovertext=sgv_today.sgv,
            hoverinfo='text'
        ),
        row=1, col=1
    )

    # carbs
    if len(carbs_today) > 1:
        agg_data = hierarchy_cluster(carbs_today, 'carbs', cutoff=cutoff_value)
        carbs_x, carbs_y = agg_data['middle_time'], agg_data[('carbs', 'sum')]
        hovertext = get_hover_data(agg_data, 'carbs', 'g')
    else:
        carbs_x, carbs_y = carbs_today.timestamp, carbs_today.carbs
        hovertext = carbs_today.carbs

    fig_timeline.add_trace(
        go.Bar(
            x=carbs_x,
            y=carbs_y,
            marker=dict(opacity=0.6),
            marker_color=colors['carbs'],
            hovertext=hovertext,
            hoverlabel=dict(namelength=0),
            name='',
            xaxis='x2',
            yaxis='y2',
            width=bar_width_factor*3600*24,
        ),
        row=2, col=1
    )

    # insulin
    if len(insulin_today) > 1:
        agg_data = hierarchy_cluster(insulin_today, 'bolus', cutoff=cutoff_value)
        bolus_x, bolus_y = agg_data['middle_time'], agg_data[('bolus', 'sum')]
        hovertext = get_hover_data(agg_data, 'bolus', 'U')
    else:
        bolus_x, bolus_y = insulin_today.timestamp, insulin_today.bolus
        hovertext = insulin_today.bolus

    fig_timeline.add_trace(
        go.Bar(
            x=bolus_x,
            y=bolus_y,
            marker=dict(opacity=0.6),
            marker_color=colors['bolus'],
            hovertext=hovertext,
            hoverlabel=dict(namelength=0),
            name='',
            xaxis='x2',
            yaxis='y3',
            width=bar_width_factor*3600*24,
        ),
        row=3, col=1
    )

    # basal
    x, y = create_basal_data(br_default_today, 'br_default')
    fig_timeline.add_trace(
        go.Scattergl(
            x=x,
            y=y,
            mode='lines',
            hoverinfo='skip',
            line=dict(color='rgba(0, 159, 219, 1)', dash='dash'),
            # connectgaps=False,
            xaxis='x2',
            yaxis='y4'
        ),
        row=4, col=1
    )

    if logs_br is not None:
        x, y = create_basal_data(br_today, 'br')
        fig_timeline.add_trace(
            go.Scattergl(
                x=x,
                y=y,
                mode='lines',
                hoverinfo='skip',
                line=dict(color='rgba(0, 159, 219, 1)'),
                fill='toself',
                fillcolor='rgba(0, 159, 219, 0.5)',
                xaxis='x2',
                yaxis='y4'
            ),
            row=4, col=1
        )
    else:
        fig_timeline.add_trace(
            go.Scattergl(
                x=x,
                y=y,
                hoverinfo='skip',
                mode='lines',
                line=dict(color='rgba(0, 159, 219, 1)'),
                fill='toself',
                fillcolor='rgba(0, 159, 219, 0.5)',
                xaxis='x2',
                yaxis='y4'
            ),
            row=4, col=1
        )

    fig_timeline.update_xaxes(type="date", autorange=False, range=x_range)
    fig_timeline.update_layout(xaxis_rangeslider_visible=False,
                               xaxis2_rangeslider_visible=False,
                               xaxis_type="date",
                               showlegend=False,
                               margin=dict(t=25, b=10, l=0, r=0),
                               plot_bgcolor='rgba(248,249,250,1)',
                               yaxis=dict(
                                   range=[0, 420],
                                   tickfont_size=8,
                                   fixedrange=True
                               ),
                               yaxis2=dict(
                                   range=[0, max(logs_carbs.carbs)],
                                   tickfont_size=8,
                                   fixedrange=True
                               ),
                               yaxis3=dict(
                                   range=[0, max(logs_insulin.bolus)],
                                   tickfont_size=8,
                                   fixedrange=True
                               ),
                               yaxis4=dict(
                                   range=[0, 1.5],
                                   tickfont_size=8,
                                   fixedrange=True
                               ),
                               font=dict(
                                   family=font,
                               ),
                               )
    fig_timeline.layout.annotations[0].update(x=0.06, font=dict(family=font, size=10))
    fig_timeline.layout.annotations[1].update(x=0.03, font=dict(family=font, size=10))
    fig_timeline.layout.annotations[2].update(x=0.03, font=dict(family=font, size=10))
    fig_timeline.layout.annotations[3].update(x=0.04, font=dict(family=font, size=10))

    return fig_timeline


def create_basal_data(logs, log_type):
    if len(logs) > 24:
        logs = logs[(logs['timestamp'].dt.hour <= logs.timestamp.iloc[-1].hour)]
    x = np.repeat(logs.timestamp, 2)
    y = np.repeat(logs[log_type], 2)
    y = np.concatenate([[0], y])
    y = y[:-1]
    y[-1] = 0
    return x, y


def hierarchy_cluster(logs_today, log_type, cutoff=1.5):
    logs_today['time'] = logs_today.timestamp.dt.hour + logs_today.timestamp.dt.minute / 60 + logs_today.timestamp.dt.second / (60 * 60)
    data = logs_today.time.to_numpy()
    nnumbers = data.reshape(-1)
    data = data.reshape(-1, 1)
    Z = scipy.cluster.hierarchy.average(data)
    color = fcluster(Z, t=cutoff, criterion='distance')
    grouped_item = pd.DataFrame(list(zip(nnumbers, color, logs_today[log_type], logs_today.timestamp)), columns=['numbers', 'segment', log_type, 'timestamp']).groupby('segment')
    agg_data = grouped_item.agg({log_type: [sum, list], 'timestamp': [min, max, list]})
    agg_data = agg_data.reset_index()
    agg_data['timedelta'] = agg_data.timestamp['max'] - agg_data.timestamp['min']
    agg_data['middle_time'] = agg_data.timestamp['min'] + agg_data['timedelta'] / 2

    return agg_data


def get_hover_data(agg_data, log_type, unit):
    hover_text = []

    for i in range(len(agg_data)):
        total_bolus = agg_data[log_type, 'sum'].iloc[i]
        dates = agg_data['timestamp', 'list'].iloc[i]
        times = [item.strftime('%H:%M') for item in dates]
        boluses = agg_data[log_type, 'list'].iloc[i]

        if len(boluses) > 1:
            bolus_list = [t + ': ' + str(b) + ' {}'.format(unit) for t, b in zip(times, boluses)]
            name = '<b>Total: ' + str(round(total_bolus, 1)) + ' {}</b>'.format(unit) + '<br />' + '<br />'.join(bolus_list)
        else:
            name = '<b>' + str(times[0]) + ': ' + str(round(total_bolus, 1)) + ' {}</b>'.format(unit)
        hover_text.append(name)
    return hover_text
