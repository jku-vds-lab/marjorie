from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import scipy
from plotly.subplots import make_subplots
from scipy.cluster import hierarchy
from scipy.cluster.hierarchy import fcluster

from colors import colors, colors_agp
from helpers import get_df_of_date, get_infos_from_group
from preprocessing import date_max, logs_sgv_plot, logs_carbs, logs_insulin
from variables import target_range, font, target_range_extended, target_range_dict

alpha_max_insulin = logs_insulin['bolus'].to_numpy().max()

max_d = 1.5

start_date = date_max - timedelta(days=1)
end_date = date_max

y_range = [-400, 450]


def get_daily_data(day):
    sgv_today = get_df_of_date(logs_sgv_plot, day)
    carbs_today = get_df_of_date(logs_carbs, day)
    insulin_today = get_df_of_date(logs_insulin, day)
    return sgv_today, carbs_today, insulin_today


def get_x_range_for_day(day):
    midnight = datetime.combine(day, datetime.min.time())
    x_range = [midnight, midnight + timedelta(days=1)]
    return x_range


def get_non_periodic_data(indices, idx):
    sgv_today = logs_sgv_plot.iloc[indices['sgv'][0][idx]:indices['sgv'][1][idx]]
    carbs_today = logs_carbs.iloc[indices['carbs'][0][idx]:indices['carbs'][1][idx]]
    insulin_today = logs_insulin.iloc[indices['insulin'][0][idx]:indices['insulin'][1][idx]]
    return sgv_today, carbs_today, insulin_today


def draw_horizon_graph(sgv_today, carbs_today, insulin_today, x_range):
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, row_heights=[0.6, 0.2, 0.2], vertical_spacing=0.05)

    sgv_today['vl'] = sgv_today['sgv'] <= target_range_dict['very low']
    sgv_today['l'] = sgv_today['sgv'].between(target_range_dict['very low'], target_range_dict['low'])
    sgv_today['h'] = sgv_today['sgv'].between(target_range_dict['high'], target_range_dict['very high'])
    sgv_today['vh'] = sgv_today['sgv'] >= target_range_dict['very high']

    # in range
    fig.add_trace(
        go.Scatter(
            x=sgv_today['timestamp'].to_list(),
            y=[0.2] * len(sgv_today),
            fill='tozeroy',
            fillcolor=colors['bg_target'],
            mode='lines',
            line=dict(color='rgba(0,0,0,0)'),
            connectgaps=False,
            hoverinfo='skip'
        ),
        row=1, col=1
    )

    # high
    sgv_high = sgv_today.loc[sgv_today['high']]
    if len(sgv_high) > 1:
        sgv_today['transform_high'] = sgv_today['sgv'].copy()
        sgv_today['transform_high'][sgv_today['vh']] = target_range_dict['very high']
        sgv_today['transform_high'][sgv_today['transform_high'] <= target_range_dict['high']] = target_range_dict['high']
        sgv_today['transform_high'] = sgv_today['transform_high'] - target_range_dict['high']
        sgv_today['transform_high'][sgv_today['sgv'] <= target_range_dict['high']] = 0
        sgv_today['transform_high'] = sgv_today['transform_high'] / (target_range_dict['very high'] - target_range_dict['high'])

        fig.add_trace(
            go.Scatter(
                x=sgv_today['timestamp'].to_list(),
                y=sgv_today['transform_high'].to_list(),
                fill='tozeroy',
                fillcolor=colors['bg_high'],
                mode='lines',
                line=dict(color='rgba(0,0,0,0)'),
                connectgaps=False,
                hoverinfo='skip'
            ),
            row=1, col=1
        )

    # low
    sgv_low = sgv_today.loc[sgv_today['low']]
    if len(sgv_low) > 1:
        sgv_today['transform_low'] = sgv_today['sgv'].copy()
        sgv_today['transform_low'][sgv_today['transform_low'] <= target_range_dict['very low']] = target_range_dict['very low']
        sgv_today['transform_low'][sgv_today['transform_low'] >= target_range_dict['low']] = target_range_dict['low']
        sgv_today['transform_low'] = sgv_today['transform_low'] - target_range_dict['low']
        sgv_today['transform_low'] = sgv_today['transform_low'] * (-1)
        sgv_today['transform_low'][sgv_today['sgv'] >= target_range_dict['low']] = 0
        sgv_today['transform_low'] = sgv_today['transform_low'] / (target_range_dict['low'] - target_range_dict['very low'])

        fig.add_trace(
            go.Scatter(
                x=sgv_today['timestamp'].to_list(),
                y=sgv_today['transform_low'].to_list(),
                fill='tozeroy',
                fillcolor=colors['bg_low'],
                mode='lines',
                line=dict(color='rgba(0,0,0,0)'),
                connectgaps=False,
                hoverinfo='skip'
            ),
            row=1, col=1
        )

    # very high
    if (sgv_today['sgv'] > target_range_dict['very high']).any():
        sgv_today['transform_very_high'] = sgv_today['sgv'].copy()
        sgv_today['transform_very_high'][sgv_today['transform_very_high'] <= target_range_dict['very high']] = target_range_dict['very high']
        sgv_today['transform_very_high'] = sgv_today['transform_very_high'] - target_range_dict['very high']
        sgv_today['transform_very_high'] = sgv_today['transform_very_high'] / (350 - target_range_dict['very high'])

        fig.add_trace(
            go.Scatter(
                x=sgv_today['timestamp'].to_list(),
                y=sgv_today['transform_very_high'].to_list(),
                fill='tozeroy',
                fillcolor=colors['bg_very_high'],
                mode='lines',
                line=dict(color='rgba(0,0,0,0)'),
                connectgaps=False,
                hoverinfo='skip'
            ),
            row=1, col=1
        )

    # very low
    if (sgv_today['sgv'] < target_range_dict['very low']).any():
        sgv_today['transform_very_low'] = sgv_today['sgv'].copy()
        sgv_today['transform_very_low'][sgv_today['transform_very_low'] >= target_range_dict['very low']] = target_range_dict['very low']
        sgv_today['transform_very_low'] = sgv_today['transform_very_low'] - target_range_dict['very low']
        sgv_today['transform_very_low'] = sgv_today['transform_very_low'] * (-1)
        sgv_today['transform_very_low'] = sgv_today['transform_very_low'] / (target_range_dict['very low'] - 40)

        fig.add_trace(
            go.Scatter(
                x=sgv_today['timestamp'].to_list(),
                y=sgv_today['transform_very_low'].to_list(),
                fill='tozeroy',
                fillcolor=colors['bg_very_low'],
                mode='lines',
                line=dict(color='rgba(0,0,0,0)'),
                connectgaps=False,
                hoverinfo='skip'
            ),
            row=1, col=1
        )

    # treatments
    if not carbs_today.empty:
        fig = plot_treatments(fig, 2, carbs_today, 'carbs', 'g', 100)
    if not insulin_today.empty:
        fig = plot_treatments(fig, 3, insulin_today, 'bolus', 'U', 10)

    x_values, x_labels = get_infos_from_group('day')
    fig.update_xaxes(type="date", range=x_range, automargin=False, visible=True, showgrid=True, tickvals=x_values, ticktext=['' for _ in x_labels])
    fig.update_layout(xaxis_rangeslider_visible=False,
                      showlegend=False,
                      width=575, height=60,
                      margin=dict(t=0, b=20, l=0, r=0),
                      plot_bgcolor=colors['background'],
                      yaxis=dict(
                          range=[0, 1],
                          tickfont_size=8,
                          visible=False
                      ),
                      yaxis2=dict(
                          range=[0, 1],
                          tickfont_size=8,
                          showgrid=False,
                          visible=False
                      ),
                      yaxis3=dict(
                          range=[0, 1],
                          tickfont_size=8,
                          showgrid=False,
                          visible=False
                      ),
                      font=dict(
                          family=font,
                      ),
                      paper_bgcolor='rgba(0,0,0,0)',
                      )

    return fig


def hierarchy_cluster(logs_today, log_type):
    logs_today['time'] = logs_today.timestamp.dt.hour + logs_today.timestamp.dt.minute / 60 + logs_today.timestamp.dt.second / (60 * 60)
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


def get_hover_data(agg_data, log_type, unit, i):
    # hover data
    total_bolus = agg_data[log_type, 'sum'].iloc[i]
    dates = agg_data['timestamp', 'list'].iloc[i]
    times = [item.strftime('%H:%M') for item in dates]
    boluses = agg_data[log_type, 'list'].iloc[i]

    if len(boluses) > 1:
        bolus_list = [t + ': ' + str(b) + ' {}'.format(unit) for t, b in zip(times, boluses)]
        name = '<b>Total: ' + str(round(total_bolus, 1)) + ' {}</b>'.format(unit) + '<br />' + '<br />'.join(bolus_list)
    else:
        name = '<b>' + str(times[0]) + ': ' + str(round(total_bolus, 1)) + ' {}</b>'.format(unit)
    return name, total_bolus


def plot_treatments(fig, row, logs_today, log_type, unit, max_value):
    if len(logs_today) > 1:

        agg_data = hierarchy_cluster(logs_today, log_type)
        for i in range(len(agg_data)):
            name, total_bolus = get_hover_data(agg_data, log_type, unit, i)

            fig.add_trace(
                go.Scatter(
                    x=[agg_data.min_time.iloc[i], agg_data.max_time.iloc[i], agg_data.max_time.iloc[i], agg_data.min_time.iloc[i]],
                    y=[0, 0, 1, 1],
                    fill='toself',
                    hoveron='fills',
                    hoverlabel=dict(font_size=9),
                    hoverinfo='text',
                    name=name,
                    line=dict(color='rgba(0,0,0,0)'),
                    fillcolor=colors[log_type],
                    opacity=min(total_bolus/max_value, 1)
                ),
                row=row, col=1
            )
    else:  # only 1 entry on that day
        min_time = logs_today.timestamp.iloc[0] - timedelta(minutes=30)
        max_time = logs_today.timestamp.iloc[0] + timedelta(minutes=30)
        name = '<b>' + str(logs_today.timestamp.iloc[0].strftime('%H:%M')) + ': ' + str(round(logs_today[log_type].iloc[0], 1)) + ' {}</b>'.format(unit)
        fig.add_trace(
            go.Scatter(
                x=[min_time, max_time, max_time, min_time],
                y=[0, 0, 1, 1],
                fill='toself',
                hoveron='fills',
                hoverinfo='text',
                hoverlabel=dict(font_size=10),
                name=name,
                line=dict(color='rgba(0,0,0,0)'),
                fillcolor=colors[log_type],
                opacity=min(logs_today[log_type].iloc[0]/max_value, 1)
            ),
            row=row, col=1
        )
    return fig


def draw_overview_daily_curve_detailed(sgv_today, carbs_today, insulin_today, x_range, box_data=None, highlight_data=None, hide_xaxis=True):
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, row_heights=[0.86, 0.07, 0.07], vertical_spacing=0.01)

    # sgv
    y_sgv = sgv_today.sgv.fillna(0)
    fig.add_trace(
        go.Scatter(
            x=sgv_today.timestamp,
            y=y_sgv,
            mode='lines',
            line=dict(color=colors['bg_target']),
            connectgaps=False,
            hovertext=sgv_today.sgv,
            hoverinfo='text'
        ),
        row=1, col=1
    )

    # below range
    sgv_low = sgv_today.loc[sgv_today['low']]
    if len(sgv_low) > 1:
        fig.add_trace(
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
            row=1, col=1
        )

    # above range
    sgv_high = sgv_today.loc[sgv_today['high']]
    if len(sgv_high) > 1:
        fig.add_trace(
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
            row=1, col=1
        )

    # target
    fig.add_trace(
        go.Scatter(
            x=sgv_today.timestamp,
            y=[target_range[1]] * len(sgv_today),
            mode='lines',
            line=dict(color='white'),
            connectgaps=False,
            hoverinfo='skip'
        ),
        row=1, col=1
    )

    fig.add_trace(
        go.Scatter(
            x=sgv_today.timestamp,
            y=[target_range[0]] * len(sgv_today),
            mode='lines',
            line=dict(color='white'),
            connectgaps=False,
            hoverinfo='skip'
        ),
        row=1, col=1
    )

    if not carbs_today.empty:
        fig = plot_treatments(fig, 2, carbs_today, 'carbs', 'g', 100)
    if not insulin_today.empty:
        fig = plot_treatments(fig, 3, insulin_today, 'bolus', 'U', 10)

    fig.update_xaxes(type="date", range=x_range, automargin=False)

    if hide_xaxis:
        fig.update_xaxes(visible=False)

    fig.update_layout(xaxis_rangeslider_visible=False,
                      # xaxis2_rangeslider_visible=False,
                      # xaxis_type="date",
                      showlegend=False,
                      width=575, height=120,
                      margin=dict(t=0, b=20, l=0, r=0),
                      plot_bgcolor=colors['background'],
                      # xaxis=dict(visible=False, showgrid=True),
                      yaxis=dict(
                          # showticklabels=False,
                          range=[30, 400],
                          tickfont_size=8,
                          visible=False
                      ),
                      yaxis2=dict(
                          range=[0, 1],
                          tickfont_size=8,
                          # overlaying="y",
                          showgrid=False,
                          visible=False
                      ),
                      yaxis3=dict(
                          range=[0, 1],
                          tickfont_size=8,
                          # overlaying="y",
                          showgrid=False,
                          visible=False
                      ),
                      font=dict(
                          family=font,
                          # size=8,
                          # color="RebeccaPurple"
                      ),
                      paper_bgcolor='rgba(0,0,0,0)',
                      )

    if box_data:  # if agp
        fig.update_xaxes(fixedrange=True, visible=False, showgrid=True)
        fig.update_layout(dragmode="select", clickmode='event+select', selectdirection='h', margin=dict(t=0, b=0, l=0, r=0))

    return fig