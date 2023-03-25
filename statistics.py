import plotly.graph_objects as go

from colors import colors
from helpers import get_tir, get_df_of_date, get_statistics, get_df_between_dates, get_mean_per_day
from preprocessing import logs_sgv, logs_carbs, logs_insulin, logs_br_default


def get_statistics_day(day):
    sgv = get_df_of_date(logs_sgv, day)['sgv']
    stats = get_statistics(sgv)
    tir = get_tir(sgv)

    carbs = get_df_of_date(logs_carbs, day)
    carbs_sum = int(carbs['carbs'].sum())

    insulin = get_df_of_date(logs_insulin, day)
    bolus_sum = int(insulin['bolus'].sum())

    basal = get_df_of_date(logs_br_default, day)
    basal_sum = int(basal['br_default'].sum())
    return stats, tir, carbs_sum, bolus_sum, basal_sum


def get_statistics_days(start_date, end_date, weekday_filter=None):
    sgv = get_df_between_dates(logs_sgv, start_date, end_date, weekday_filter)['sgv']
    stats = get_statistics(sgv)
    tir = get_tir(sgv)

    carbs = get_df_between_dates(logs_carbs, start_date, end_date, weekday_filter)
    carbs_sum = int(get_mean_per_day(carbs, 'carbs'))

    insulin = get_df_between_dates(logs_insulin, start_date, end_date, weekday_filter)
    bolus_sum = get_mean_per_day(insulin, 'bolus').iloc[0]

    basal = get_df_between_dates(logs_br_default, start_date, end_date, weekday_filter)
    basal_sum = get_mean_per_day(basal, 'br_default').iloc[0]

    return stats, tir, carbs_sum, bolus_sum, basal_sum


def get_tir_plot(tir):

    layout = go.Layout(width=325, height=80, margin=dict(t=0, b=0, l=0, r=0), plot_bgcolor='rgba(248,249,250,1)',)
    fig = go.Figure(layout=layout)

    color_names = ['bg_very_low', 'bg_low', 'bg_target', 'bg_high', 'bg_very_high']

    for i in range(len(tir)):
        if tir[i] == 0:
            color = colors['background']
        else:
            color = colors[color_names[i]]

        fig.add_trace(
            go.Bar(
                name=color_names[i],
                x=[tir[i]],
                y=[0],
                hoverinfo='skip',
                orientation='h',
                # base=base,
                # offsetgroup=0,
                # width=1.5,
                marker_color=color
            ))
        # base += max(5, tir[i])

    fig.update_layout(
        showlegend=False,
        yaxis=dict(
            showticklabels=False,
            showgrid=False,
            range=[-1, 1]
        ),
        xaxis=dict(
            showticklabels=False,
            showgrid=False,
            range=[0, 100],
        ),
        barmode='stack',
        margin=dict(t=0, b=0, l=0, r=0),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
    )
    return fig

