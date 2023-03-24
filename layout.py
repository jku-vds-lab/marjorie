from datetime import timedelta, datetime, date
import json
import dash
import dash_bootstrap_components as dbc
from dash import dcc, html, ctx
from dash_extensions.enrich import Output, DashProxy, Input, MultiplexerTransform, State, callback_context, ALL
from aggregations import draw_seasonal_graph_day, draw_seasonal_graph, draw_agp_carbs_bolus, draw_full_agp, agp_xaxis
from colors import colors_patterns, colors_heatmap, targets_heatmap, get_prebolus_button_color
from daily import draw_daily_plot
import numpy as np
from insights import draw_pattern_overview, add_time_before_after, get_insight_dataset, get_insight_clusters, draw_hierarchical_pattern_overview, get_logs_meals, get_logs_from_indices, get_insight_data_meals, \
    get_time_of_day_from_number, get_curve_overview_plot, get_dataset, filter_function_time_of_day, filter_function_meal_size, get_insight_data_hypos
from motif_detection import get_fetures_sliding_windows, draw_clustering
from pattern_detail import draw_pattern_detail_plot, get_daily_data, get_x_range_for_day, get_non_periodic_data, draw_pattern_detail_plot_curve
from preprocessing import dates, logs_sgv, date_max, date_min, start_date, end_date, sgv_array_for_agp, date_dict, logs_carbs, logs_insulin, logs_br_default, start_date_insights
from statistics import get_tir_plot, get_statistics_day, get_statistics_days
from helpers import convert_datestring, get_df_between_dates, get_tir, get_statistics, check_timebox, get_log_indices, calculate_tir_time, get_mean_per_day, get_df_of_date
from variables import num_horizon_graphs, n_filters, initial_number_of_days, num_insight_details, time_before_meal, time_after_meal
from assets.styles import *
import re
from datetime import datetime, timedelta
import warnings
from pandas.core.common import SettingWithCopyWarning

warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)

# Connect to main app.py file
FONT_AWESOME = "https://use.fontawesome.com/releases/v5.7.2/css/all.css"
app = DashProxy(prevent_initial_callbacks=True, transforms=[MultiplexerTransform()], suppress_callback_exceptions=True, external_stylesheets=[dbc.themes.BOOTSTRAP, FONT_AWESOME])

################################################################################
# GENERAL LAYOUT
################################################################################

content = html.Div(id="page-content", style=CONTENT_STYLE)
app.layout = html.Div([
    dcc.Location(id='url'),
    dbc.Row([
        dbc.Col(
            ################################################################################
            # SIDEBAR
            ################################################################################
            html.Div(
                [
                    html.H2("", className="display-4"),
                    html.Hr(),
                    dbc.Nav(
                        [
                            dbc.NavLink("Summary", href="/agp", active="exact"),
                            dbc.NavLink("Diary", href="/daily", active="exact"),
                            dbc.NavLink("Insights", href="/insights", active="exact"),
                        ],
                        vertical=True,
                        pills=True,
                    ),
                ],
                style=SIDEBAR_STYLE_LEFT,
            )
            , width=1),
        ################################################################################
        # CONTENT
        ################################################################################
        dbc.Col(content, width=11),
    ]),
    dcc.Store(id='memory_date_picker_range_clicked', data=1),
    dcc.Store(id='memory_agp_initial_zoom', data=1),
    dcc.Store(id='cache',
              data=json.dumps({'clusters': None,
                               'n_clusters': None,
                               'clicked_date': date_max.strftime('%d/%m/%Y'),
                               })
              )
])

################################################################################
# DAILY SECTION
################################################################################

# statistic calculations
stats, tir, carbs_sum, bolus_sum, basal_sum = get_statistics_day(date_max.date())

layout_daily = html.Div(
    dbc.Row([
        dbc.Col(
            ################################################################################
            # DAILY GRAPH
            ################################################################################
            children=dbc.Card(
                [
                    dbc.CardBody(
                        dcc.Graph(
                            figure=draw_daily_plot(date_max.date()),
                            id='daily_graph'
                        ),
                    )
                ]
            ),
            width=8,
            md=8,
            xs=12
        ),
        dbc.Col(children=
        [
            ################################################################################
            # DATE PICKER DAILY
            ################################################################################
            dbc.Row([
                dbc.Col(html.Button(children=html.Span([html.I(className="fas fa-caret-left fa-2x", style={'position': 'relative', 'left': '-4px'})]),
                                    id='date_daily_back',
                                    n_clicks_timestamp=0,
                                    style=buttons_style_icon
                                    ),
                        width=3),
                dbc.Col(dcc.DatePickerSingle(
                    id='date_picker_daily',
                    min_date_allowed=date_min.date(),
                    max_date_allowed=date_max.date(),
                    initial_visible_month=date_max.date(),
                    date=date_max.date(),
                    display_format='DD/MM/YYYY',
                ), style={'padding': '0 0 0 5.2%'}, width=6),
                dbc.Col(html.Button(children=html.Span([html.I(className="fas fa-caret-right fa-2x")]),
                                    id='date_daily_forward',
                                    n_clicks_timestamp=0,
                                    style=buttons_style_icon
                                    ),
                        width=3)
            ],
                style={'padding': '0em 0em 1em 0em'}
            ),
            ################################################################################
            # STATISTICS DAILY
            ################################################################################
            html.Div(
                [
                    ################################################################################
                    # BASIC METRICS
                    ################################################################################
                    dbc.Row(
                        [
                            dbc.Col(
                                dbc.Card(
                                    [
                                        dbc.Row(html.Div(children='eA1c', className='c-stats-card-tir__title')),
                                        html.Div(
                                            [
                                                html.Div(children=str(round(stats['ea1c'], 1)), id='stats_daily_sgv_ea1c', className='c-stats-card__value'),
                                                html.Div(children=' %', className='c-stats-card-tir__unit'),
                                            ],
                                            style={'text-align': 'center', 'padding': '12% 0'}
                                        ),
                                    ],
                                    style={'height': '10rem', 'width': '10rem'},
                                    className='image-border',
                                ),
                            ),
                            dbc.Col(
                                dbc.Card(
                                    [
                                        dbc.Row(html.Div(children='Average', className='c-stats-card-tir__title')),
                                        html.Div(
                                            [
                                                html.Div(children=str(int(stats['mean'])), id='stats_daily_sgv_mean', className='c-stats-card__value'),
                                                html.Div(children=' mg/dL', className='c-stats-card-tir__unit'),
                                            ],
                                            style={'text-align': 'center', 'padding': '12% 0'}
                                        ),
                                    ],
                                    style={'height': '10rem', 'width': '10rem'},
                                    className='image-border',
                                ),
                            ),
                            dbc.Col(
                                dbc.Card(
                                    [
                                        dbc.Row(html.Div(children='CV', className='c-stats-card-tir__title')),
                                        html.Div(
                                            [
                                                html.Div(children=str(int(stats['std'] / stats['mean'] * 100)), id='stats_daily_sgv_std', className='c-stats-card__value'),
                                                html.Div(children=' %', className='c-stats-card-tir__unit'),
                                            ],
                                            style={'text-align': 'center', 'padding': '12% 0'}
                                        ),
                                    ],
                                    style={'height': '10rem', 'width': '10rem'},
                                    className='image-border',
                                ),
                            )
                        ]
                    ),
                    ################################################################################
                    # TIME IN RANGE
                    ################################################################################
                    html.Div(children='Time in Range', style={'font-size': 'small', 'padding': '3% 3%', 'font-weight': 'bold'}),
                    dbc.Row(
                        [
                            dbc.Col(
                                [
                                    dbc.Card(
                                        [
                                            dbc.Row(html.Div(children='Low', className='c-stats-card-tir__title')),
                                            dbc.Row(html.Div(children='54 - 70 mg/dl', className='c-stats-card-tir__range')),
                                            html.Div(
                                                [
                                                    html.Div(children=str(tir[1]), id='stats_daily_low', className='c-stats-card-tir__value-x', style={'color': colors['bg_low']}),
                                                    html.Div(children=' %', className='c-stats-card-tir__unit'),
                                                ],
                                                className='c-stats-card-tir__wrapper'
                                            ),
                                            dbc.Row(html.Div(children=calculate_tir_time(tir[1]), id='stats_daily_low_time', className='c-stats-card-tir__time', style={'top': '-2rem', 'color': colors['bg_low']})),
                                        ],
                                        style={'height': '8rem', 'width': '10rem'},
                                        className='image-border-top',
                                    ),
                                    dbc.Card(
                                        [
                                            dbc.Row(html.Div(children='Very Low', className='c-stats-card-tir__title')),
                                            dbc.Row(html.Div(children='< 54 mg/dl', className='c-stats-card-tir__range')),
                                            html.Div(
                                                [
                                                    html.Div(children=str(tir[0]), id='stats_daily_very_low', className='c-stats-card-tir__value-x', style={'color': colors['bg_very_low']}),
                                                    html.Div(children=' %', className='c-stats-card-tir__unit'),
                                                ],
                                                className='c-stats-card-tir__wrapper'
                                            ),
                                            dbc.Row(
                                                html.Div(children=calculate_tir_time(tir[0]), id='stats_daily_very_low_time', className='c-stats-card-tir__time', style={'top': '-2rem', 'color': colors['bg_very_low']})),
                                        ],
                                        style={'height': '8rem', 'width': '10rem'},
                                        className='image-border-bottom',
                                    )
                                ]
                            ),
                            dbc.Col(
                                dbc.Card(
                                    [
                                        dbc.Row(html.Div(children='Target', className='c-stats-card-tir__title')),
                                        dbc.Row(html.Div(children='70 - 180 mg/dl', className='c-stats-card-tir__range')),
                                        html.Div(
                                            [
                                                html.Div(children=str(tir[2]), id='stats_daily_target', className='c-stats-card-tir__value-xx', style={'color': colors['bg_target']}),
                                                html.Div(children=' %', className='c-stats-card-tir__unit'),
                                            ],
                                            style={'text-align': 'center', 'padding': '15% 0'}
                                        ),
                                        dbc.Row(html.Div(children=calculate_tir_time(tir[2]), id='stats_daily_target_time', className='c-stats-card-tir__time', style={'color': colors['bg_target'], })),
                                    ],
                                    style={'height': '16rem', 'width': '10rem'},
                                    className='image-border',
                                )
                            ),
                            dbc.Col(
                                [
                                    dbc.Card(
                                        [
                                            dbc.Row(html.Div(children='High', className='c-stats-card-tir__title')),
                                            dbc.Row(html.Div(children='180 - 250 mg/dl', className='c-stats-card-tir__range')),
                                            html.Div(
                                                [
                                                    html.Div(children=str(tir[3]), id='stats_daily_high', className='c-stats-card-tir__value-x', style={'color': colors['bg_high']}),
                                                    html.Div(children=' %', className='c-stats-card-tir__unit'),
                                                ],
                                                className='c-stats-card-tir__wrapper'
                                            ),
                                            dbc.Row(html.Div(children=calculate_tir_time(tir[3]), id='stats_daily_high_time', className='c-stats-card-tir__time',
                                                             style={'text-align': 'center', 'top': '-2rem', 'color': colors['bg_high']})),
                                        ],
                                        style={'height': '8rem', 'width': '10rem'},
                                        className='image-border-top',
                                    ),
                                    dbc.Card(
                                        [
                                            html.Div(children='Very High', className='c-stats-card-tir__title'),
                                            html.Div(children='> 250 mg/dl', className='c-stats-card-tir__range'),
                                            html.Div(
                                                [
                                                    html.Div(children=str(tir[4]), id='stats_daily_very_high', className='c-stats-card-tir__value-x', style={'color': colors['bg_very_high']}),
                                                    html.Div(children=' %', className='c-stats-card-tir__unit'),
                                                ],
                                                className='c-stats-card-tir__wrapper'
                                            ),
                                            dbc.Row(html.Div(children=calculate_tir_time(tir[4]), id='stats_daily_very_high_time', className="c-stats-card-tir__time",
                                                             style={'top': '-2rem', 'color': colors['bg_very_high']})),
                                        ],
                                        style={'height': '8rem', 'width': '10rem'},
                                        className='image-border-bottom',
                                    )
                                ]
                            ),
                        ]
                    ),
                    html.Div(
                        dcc.Graph(
                            figure=get_tir_plot(tir),
                            id='stats_daily_tir_graph',
                            style={'padding': '0rem 0rem 0rem 0.5rem'},
                            config={
                                'displayModeBar': False
                            }
                        ),
                        style={'border-radius': '25px 25px 25px 25px', 'overflow': 'hidden'}
                    ),

                    ################################################################################
                    # TREATMENTS
                    ################################################################################
                    html.Div(children='Treatments', style={'font-size': 'small', 'padding': '0% 3% 3%', 'font-weight': 'bold'}),
                    dbc.Row(
                        [
                            dbc.Col(
                                dbc.Card(
                                    [
                                        dbc.Row(html.Div(children='Carbs', className='c-stats-card-tir__title')),
                                        html.Div(
                                            [
                                                html.Div(children=carbs_sum, id='stats_daily_carbs', className='c-stats-card__value', style={'color': colors['carbs']}),
                                                html.Div(children=' g', className='c-stats-card-tir__unit'),
                                            ],
                                            style={'text-align': 'center', 'padding': '12% 0'}
                                        ),
                                    ],
                                    style={'height': '10rem', 'width': '10rem'},
                                    className='image-border',
                                ),
                            ),
                            dbc.Col(
                                dbc.Card(
                                    [
                                        dbc.Row(html.Div(children='Insulin ({} U)'.format(bolus_sum + basal_sum), className='c-stats-card-tir__title')),
                                        html.Div(
                                            dbc.Row(
                                                [
                                                    dbc.Col(
                                                        [
                                                            html.Div(
                                                                [
                                                                    html.Div(children=bolus_sum, id='stats_daily_bolus', className='c-stats-card__value', style={'color': colors['bolus'], }),
                                                                    html.Div(children=' U', className='c-stats-card-tir__unit'),
                                                                ],
                                                                style={'text-align': 'center', 'padding': '12% 0'}
                                                            ),
                                                            dbc.Row(html.Div(children=str(round(bolus_sum / (bolus_sum + basal_sum) * 100)) + ' %',
                                                                             className='c-stats-card-treat__prctg',
                                                                             style={'top': '-2rem', 'color': colors['bolus']})),
                                                        ],
                                                        style={'border-right': '1px solid', 'border-color': 'LightGray'}
                                                    ),
                                                    dbc.Col(
                                                        [
                                                            html.Div(
                                                                [
                                                                    html.Div(children=basal_sum, id='stats_daily_basal', className='c-stats-card__value', style={'color': colors['basal'], }),
                                                                    html.Div(children=' U', className='c-stats-card-tir__unit'),
                                                                ],
                                                                style={'text-align': 'center', 'padding': '12% 0'}
                                                            ),
                                                            dbc.Row(
                                                                html.Div(children=str(round(basal_sum / (bolus_sum + basal_sum) * 100)) + ' %',
                                                                         className='c-stats-card-treat__prctg',
                                                                         style={'top': '-2rem', 'color': colors['basal']})),
                                                        ]
                                                    ),
                                                ],
                                                className="h-75",
                                            ),
                                        ),

                                    ],
                                    style={'height': '10rem', 'width': '22rem'},
                                    className='image-border',
                                ),
                                width=8
                            )
                        ]
                    )
                ]
            )
        ],
            width=4, md=4, className="sm-hide")
    ])
)

################################################################################
# AGP SECTION
################################################################################

# statistic calculations
stats, tir, carbs_sum, bolus_sum, basal_sum = get_statistics_days(start_date, end_date)

layout_agp = html.Div(
    [
        dbc.Row(
            [
                dbc.Col(
                    children=[
                        ################################################################################
                        # DATE SELECTOR MENU
                        ################################################################################
                        html.Div([
                            dbc.Row(
                                [
                                    ###############################################################################
                                    # DATE PICKER
                                    ###############################################################################
                                    dbc.Col(
                                        dbc.Row([
                                            dcc.DatePickerRange(
                                                id='agp_date-picker-range',
                                                min_date_allowed=date_min.date(),
                                                max_date_allowed=date_max.date(),
                                                initial_visible_month=date_max.date(),
                                                start_date=start_date.date(),
                                                end_date=end_date.date(),
                                                style={'z-index': '9000'}
                                            )
                                        ]), width=6),
                                    ################################################################################
                                    # QUICK DATE
                                    ################################################################################
                                    dbc.Col(
                                        dbc.ButtonGroup(
                                            [dbc.Button("2w", id={'type': 'agp_quick_date_button', 'index': 2}, outline=True, color="secondary", n_clicks=0),
                                             dbc.Button("4w", id={'type': 'agp_quick_date_button', 'index': 4}, outline=True, color="secondary", n_clicks=0),
                                             dbc.Button("8w", id={'type': 'agp_quick_date_button', 'index': 8}, outline=True, color="secondary", n_clicks=0)],
                                            size="md",
                                            # style={'position': 'relative', 'left': '-5em'}
                                        ),
                                        width=2,
                                    ),
                                    ################################################################################
                                    # WEEKDAY SELECTOR
                                    ################################################################################
                                    dbc.Col(
                                        [
                                            dbc.Col([
                                                dbc.Button(weekday,
                                                           color="secondary",
                                                           className="me-1",
                                                           size="sm",
                                                           style={'height': '2rem'},
                                                           active=True,
                                                           outline=True,
                                                           n_clicks=0,
                                                           id='agp_weekday_button-{}-'.format(i)
                                                           )
                                                for i, weekday in enumerate(['Mo', 'Tu', 'We', 'Th', 'Fr', 'Sa', 'Su'])
                                            ]),
                                            dbc.Row(dbc.Button("Show weekday statistics",
                                                               id='agp_open_weekly_stats',
                                                               n_clicks=0,
                                                               color="link",
                                                               size="sm",
                                                               style=style_link)
                                                    ),
                                            dbc.Modal(
                                                [
                                                    dbc.ModalHeader(dbc.ModalTitle("Weekdays")),
                                                    dbc.ModalBody(
                                                        html.Div([
                                                            # html.Div(children='Weekdays', style={'font-family': font, 'font-size': 'small', 'padding': '5% 0% 5% 0%'}),
                                                            dcc.Graph(
                                                                id='agp_week_graph',
                                                                figure=draw_seasonal_graph('week', start_date, end_date)
                                                            ),
                                                            # weekday_checklist
                                                        ])
                                                    ),
                                                ],
                                                id="agp_modal",
                                                is_open=False,
                                                style={'z-index': '11000'}
                                            ),
                                        ]
                                    )
                                ]
                            )
                        ]),
                        ################################################################################
                        # GRAPH
                        ################################################################################
                        html.Div(
                            dbc.Card(
                                children=
                                [
                                    # dbc.CardHeader('AGP'),
                                    dbc.CardBody(
                                        dcc.Graph(
                                            id='agp_graph',
                                            figure=draw_full_agp(start_date, end_date),
                                            config={
                                                'displayModeBar': False
                                            }
                                        )
                                    )
                                ],
                                className='image-border',
                                style={'padding': '0 2em'}
                            ),
                            style={'margin': '2rem 0rem'}
                        ),
                        html.Div(
                            children=dcc.Graph(
                                figure=agp_xaxis(width=620, margin_left=5)
                            ),
                            style={'position': 'relative', 'top': '-2.5em', 'left': '-0.5em', 'padding': '0rem 0rem 0rem 60px'}
                        ),
                        dbc.Button('Explore days in detail', id='agp_explore_button', n_clicks=0, outline=True, color='secondary')
                    ],
                    width=8,
                ),
                dbc.Col(
                    html.Div(
                        children=[
                            ################################################################################
                            # STATISTICS AGP
                            ################################################################################
                            html.Div(
                                [
                                    ################################################################################
                                    # BASIC METRICS
                                    ################################################################################
                                    html.Br(),
                                    dbc.Row(
                                        [
                                            dbc.Col(
                                                dbc.Card(
                                                    [
                                                        dbc.Row(html.Div(children='eA1c', className='c-stats-card-tir__title')),
                                                        html.Div(
                                                            [
                                                                html.Div(children=str(round(stats['ea1c'], 1)), id='agp_stats_sgv_ea1c', className='c-stats-card__value'),
                                                                html.Div(children=' %', className='c-stats-card-tir__unit'),
                                                            ],
                                                            style={'text-align': 'center', 'padding': '12% 0'}
                                                        ),
                                                    ],
                                                    style={'height': '10rem', 'width': '10rem'},
                                                    className='image-border',
                                                ),
                                            ),
                                            dbc.Col(
                                                dbc.Card(
                                                    [
                                                        dbc.Row(html.Div(children='Average', className='c-stats-card-tir__title')),
                                                        html.Div(
                                                            [
                                                                html.Div(children=str(int(stats['mean'])), id='agp_stats_sgv_mean', className='c-stats-card__value'),
                                                                html.Div(children=' mg/dL', className='c-stats-card-tir__unit'),
                                                            ],
                                                            style={'text-align': 'center', 'padding': '12% 0'}
                                                        ),
                                                    ],
                                                    style={'height': '10rem', 'width': '10rem'},
                                                    className='image-border',
                                                ),
                                            ),
                                            dbc.Col(
                                                dbc.Card(
                                                    [
                                                        dbc.Row(html.Div(children='CV', className='c-stats-card-tir__title')),
                                                        html.Div(
                                                            [
                                                                html.Div(children=str(int(stats['std'] / stats['mean'] * 100)), id='agp_stats_sgv_std', className='c-stats-card__value'),
                                                                html.Div(children=' %', className='c-stats-card-tir__unit'),
                                                            ],
                                                            style={'text-align': 'center', 'padding': '12% 0'}
                                                        ),
                                                    ],
                                                    style={'height': '10rem', 'width': '10rem'},
                                                    className='image-border',
                                                ),
                                            )
                                        ]
                                    ),
                                    ################################################################################
                                    # TIME IN RANGE
                                    ################################################################################
                                    html.Div(children='Time in Range', style={'font-size': 'small', 'padding': '3% 3%', 'font-weight': 'bold'}),
                                    dbc.Row(
                                        [
                                            dbc.Col(
                                                [
                                                    dbc.Card(
                                                        [
                                                            dbc.Row(html.Div(children='Low', className='c-stats-card-tir__title')),
                                                            dbc.Row(html.Div(children='54 - 70 mg/dl', className='c-stats-card-tir__range')),
                                                            html.Div(
                                                                [
                                                                    html.Div(children=str(tir[1]), id='agp_stats_low', className='c-stats-card-tir__value-x', style={'color': colors['bg_low']}),
                                                                    html.Div(children=' %', className='c-stats-card-tir__unit'),
                                                                ],
                                                                className='c-stats-card-tir__wrapper'
                                                            ),
                                                            dbc.Row(html.Div(children=calculate_tir_time(tir[1]), id='agp_stats_low_time', className='c-stats-card-tir__time',
                                                                             style={'top': '-2rem', 'color': colors['bg_low']})),
                                                        ],
                                                        style={'height': '8rem', 'width': '10rem'},
                                                        className='image-border-top',
                                                    ),
                                                    dbc.Card(
                                                        [
                                                            dbc.Row(html.Div(children='Very Low', className='c-stats-card-tir__title')),
                                                            dbc.Row(html.Div(children='< 54 mg/dl', className='c-stats-card-tir__range')),
                                                            html.Div(
                                                                [
                                                                    html.Div(children=str(tir[0]), id='agp_stats_very_low', className='c-stats-card-tir__value-x', style={'color': colors['bg_very_low']}),
                                                                    html.Div(children=' %', className='c-stats-card-tir__unit'),
                                                                ],
                                                                className='c-stats-card-tir__wrapper'
                                                            ),
                                                            dbc.Row(
                                                                html.Div(children=calculate_tir_time(tir[0]), id='agp_stats_very_low_time', className='c-stats-card-tir__time',
                                                                         style={'top': '-2rem', 'color': colors['bg_very_low']})),
                                                        ],
                                                        style={'height': '8rem', 'width': '10rem'},
                                                        className='image-border-bottom',
                                                    )
                                                ]
                                            ),
                                            dbc.Col(
                                                dbc.Card(
                                                    [
                                                        dbc.Row(html.Div(children='Target', className='c-stats-card-tir__title')),
                                                        dbc.Row(html.Div(children='70 - 180 mg/dl', className='c-stats-card-tir__range')),
                                                        html.Div(
                                                            [
                                                                html.Div(children=str(tir[2]), id='agp_stats_target', className='c-stats-card-tir__value-xx', style={'color': colors['bg_target']}),
                                                                html.Div(children=' %', className='c-stats-card-tir__unit'),
                                                            ],
                                                            style={'text-align': 'center', 'padding': '15% 0'}
                                                        ),
                                                        dbc.Row(html.Div(children=calculate_tir_time(tir[2]), id='agp_stats_target_time', className='c-stats-card-tir__time', style={'color': colors['bg_target'], })),
                                                    ],
                                                    style={'height': '16rem', 'width': '10rem'},
                                                    className='image-border',
                                                )
                                            ),
                                            dbc.Col(
                                                [
                                                    dbc.Card(
                                                        [
                                                            dbc.Row(html.Div(children='High', className='c-stats-card-tir__title')),
                                                            dbc.Row(html.Div(children='180 - 250 mg/dl', className='c-stats-card-tir__range')),
                                                            html.Div(
                                                                [
                                                                    html.Div(children=str(tir[3]), id='agp_stats_high', className='c-stats-card-tir__value-x', style={'color': colors['bg_high']}),
                                                                    html.Div(children=' %', className='c-stats-card-tir__unit'),
                                                                ],
                                                                className='c-stats-card-tir__wrapper'
                                                            ),
                                                            dbc.Row(html.Div(children=calculate_tir_time(tir[3]), id='agp_stats_high_time', className='c-stats-card-tir__time',
                                                                             style={'text-align': 'center', 'top': '-2rem', 'color': colors['bg_high']})),
                                                        ],
                                                        style={'height': '8rem', 'width': '10rem'},
                                                        className='image-border-top',
                                                    ),
                                                    dbc.Card(
                                                        [
                                                            html.Div(children='Very High', className='c-stats-card-tir__title'),
                                                            html.Div(children='> 250 mg/dl', className='c-stats-card-tir__range'),
                                                            html.Div(
                                                                [
                                                                    html.Div(children=str(tir[4]), id='agp_stats_very_high', className='c-stats-card-tir__value-x', style={'color': colors['bg_very_high']}),
                                                                    html.Div(children=' %', className='c-stats-card-tir__unit'),
                                                                ],
                                                                className='c-stats-card-tir__wrapper'
                                                            ),
                                                            dbc.Row(html.Div(children=calculate_tir_time(tir[4]), id='agp_stats_very_high_time', className="c-stats-card-tir__time",
                                                                             style={'top': '-2rem', 'color': colors['bg_very_high']})),
                                                        ],
                                                        style={'height': '8rem', 'width': '10rem'},
                                                        className='image-border-bottom',
                                                    )
                                                ]
                                            ),
                                        ]
                                    ),
                                    html.Div(
                                        dcc.Graph(
                                            figure=get_tir_plot(tir),
                                            id='agp_stats_tir_graph',
                                            style={'padding': '0rem 0rem 0rem 0.5rem'},
                                            config={
                                                'displayModeBar': False
                                            }
                                        ),
                                        style={'border-radius': '25px 25px 25px 25px', 'overflow': 'hidden'}
                                    ),

                                    ################################################################################
                                    # TREATMENTS
                                    ################################################################################
                                    html.Div(children='Treatments', style={'font-size': 'small', 'padding': '0% 3% 3%', 'font-weight': 'bold'}),
                                    dbc.Row(
                                        [
                                            dbc.Col(
                                                dbc.Card(
                                                    [
                                                        dbc.Row(html.Div(children='Carbs', className='c-stats-card-tir__title')),
                                                        html.Div(
                                                            [
                                                                html.Div(children=carbs_sum, id='agp_stats_carbs', className='c-stats-card__value', style={'color': colors['carbs']}),
                                                                html.Div(children=' g', className='c-stats-card-tir__unit'),
                                                            ],
                                                            style={'text-align': 'center', 'padding': '12% 0'}
                                                        ),
                                                    ],
                                                    style={'height': '10rem', 'width': '10rem'},
                                                    className='image-border',
                                                ),
                                            ),
                                            dbc.Col(
                                                dbc.Card(
                                                    [
                                                        dbc.Row(html.Div(children='Insulin ({} U)'.format(bolus_sum + basal_sum), className='c-stats-card-tir__title')),
                                                        html.Div(
                                                            dbc.Row(
                                                                [
                                                                    dbc.Col(
                                                                        [
                                                                            html.Div(
                                                                                [
                                                                                    html.Div(children=bolus_sum, id='agp_stats_bolus', className='c-stats-card__value', style={'color': colors['bolus'], }),
                                                                                    html.Div(children=' U', className='c-stats-card-tir__unit'),
                                                                                ],
                                                                                style={'text-align': 'center', 'padding': '12% 0'}
                                                                            ),
                                                                            dbc.Row(html.Div(children=str(round(bolus_sum / (bolus_sum + basal_sum) * 100)) + ' %',
                                                                                             className='c-stats-card-treat__prctg',
                                                                                             style={'top': '-2rem', 'color': colors['bolus']})),
                                                                        ],
                                                                        style={'border-right': '1px solid', 'border-color': 'LightGray'}
                                                                    ),
                                                                    dbc.Col(
                                                                        [
                                                                            html.Div(
                                                                                [
                                                                                    html.Div(children=basal_sum, id='agp_stats_basal', className='c-stats-card__value', style={'color': colors['basal'], }),
                                                                                    html.Div(children=' U', className='c-stats-card-tir__unit'),
                                                                                ],
                                                                                style={'text-align': 'center', 'padding': '12% 0'}
                                                                            ),
                                                                            dbc.Row(
                                                                                html.Div(children=str(round(basal_sum / (bolus_sum + basal_sum) * 100)) + ' %',
                                                                                         className='c-stats-card-treat__prctg',
                                                                                         style={'top': '-2rem', 'color': colors['basal']})),
                                                                        ]
                                                                    ),
                                                                ],
                                                                className="h-75",
                                                            ),
                                                        ),

                                                    ],
                                                    style={'height': '10rem', 'width': '22rem'},
                                                    className='image-border',
                                                ),
                                                width=8
                                            )
                                        ]
                                    )
                                ]
                            )
                        ],
                        style={'position': 'fixed', 'top': '0', 'z-index': '900', 'height': '100%', "background-color": "#f8f9fa"}
                    ),
                    width=4,
                ),
            ]
        )
    ]
)

days_horizon_graphs = [date for date in dates if (start_date.date() <= date <= end_date.date())]
layout_overview = html.Div(
    [
        dbc.Row(
            [
                dbc.Col(
                    children=[
                        ################################################################################
                        # DATE SELECTOR MENU
                        ################################################################################
                        html.Div([
                            dbc.Row(
                                [
                                    ###############################################################################
                                    # DATE PICKER
                                    ###############################################################################
                                    dbc.Col(
                                        dbc.Row([
                                            dcc.DatePickerRange(
                                                id='overview_date-picker-range',
                                                min_date_allowed=date_min.date(),
                                                max_date_allowed=date_max.date(),
                                                initial_visible_month=date_max.date(),
                                                start_date=start_date.date(),
                                                end_date=end_date.date(),
                                                style={'z-index': '9000'}
                                            )
                                        ]), width=6),
                                    ################################################################################
                                    # QUICK DATE
                                    ################################################################################
                                    dbc.Col(
                                        dbc.ButtonGroup(
                                            [dbc.Button("2w", id={'type': 'overview_quick_date_button', 'index': 2}, outline=True, color="secondary", n_clicks=0),
                                             dbc.Button("4w", id={'type': 'overview_quick_date_button', 'index': 4}, outline=True, color="secondary", n_clicks=0),
                                             dbc.Button("8w", id={'type': 'overview_quick_date_button', 'index': 8}, outline=True, color="secondary", n_clicks=0)],
                                            size="md",
                                            # style={'position': 'relative', 'left': '-5em'}
                                        ),
                                        width=2,
                                    ),
                                    ################################################################################
                                    # WEEKDAY SELECTOR
                                    ################################################################################
                                    dbc.Col(
                                        [
                                            dbc.Col([
                                                dbc.Button(weekday,
                                                           color="secondary",
                                                           className="me-1",
                                                           size="sm",
                                                           style={'height': '2rem'},
                                                           active=True,
                                                           outline=True,
                                                           n_clicks=0,
                                                           id='overview_weekday_button-{}-'.format(i))
                                                for i, weekday in enumerate(['Mo', 'Tu', 'We', 'Th', 'Fr', 'Sa', 'Su'])
                                            ]),
                                            dbc.Row(dbc.Button("Show weekday statistics",
                                                               id='overview_open_weekly_stats',
                                                               n_clicks=0,
                                                               color="link",
                                                               size="sm",
                                                               style=style_link)
                                                    ),
                                            dbc.Modal(
                                                [
                                                    dbc.ModalHeader(dbc.ModalTitle("Weekdays")),
                                                    dbc.ModalBody(
                                                        html.Div([
                                                            # html.Div(children='Weekdays', style={'font-family': font, 'font-size': 'small', 'padding': '5% 0% 5% 0%'}),
                                                            dcc.Graph(
                                                                id='overview_week_graph',
                                                                figure=draw_seasonal_graph('week', start_date, end_date)
                                                            ),
                                                            # weekday_checklist
                                                        ])
                                                    ),
                                                ],
                                                id="overview_modal",
                                                is_open=False,
                                                style={'z-index': '11000'}
                                            ),
                                        ]
                                    )
                                ]
                            )
                        ]),
                        ################################################################################
                        # AGP GRAPH
                        ################################################################################
                        html.Div([
                            html.Div(
                                dbc.Card(
                                    children=
                                    [
                                        dbc.CardBody(
                                            dcc.Graph(
                                                id='overview_agp_graph',
                                                figure=draw_seasonal_graph_day(start_date, end_date),
                                                config={
                                                    'displayModeBar': False
                                                }
                                            )
                                        )
                                    ],
                                    style={'padding': '0rem 0rem 0rem 60px'},  # ice: 6rem
                                    className='image-border',
                                    # color=colors['background']
                                ),
                                style={'position': 'sticky', 'top': '0', 'z-index': '900'},
                            ),
                            ################################################################################
                            # TIME AXIS
                            ################################################################################
                            html.Div(
                                children=dcc.Graph(
                                    figure=agp_xaxis()
                                ),
                                style={'position': 'fixed', 'bottom': '0', 'z-index': '9000', 'padding': '0rem 0rem 0rem 60px'}
                            ),
                            ################################################################################
                            # HORIZON GRAPHS
                            ################################################################################
                            dbc.Card(
                                id='overview_horizon_graphs',
                                children=[
                                             html.Div(
                                                 id='pattern_detail_agp_div_{}'.format(i),
                                                 children=
                                                 dbc.Row(
                                                     [
                                                         dbc.Col(
                                                             [
                                                                 html.Button(children=days_horizon_graphs[i].strftime('%a, %d/%m'),
                                                                             # children=days_pattern_detail[i].strftime('%a, %d/%m'),
                                                                             id='btn_agp_date-{}-'.format(i),
                                                                             n_clicks=0,
                                                                             style=buttons_style_agp_date
                                                                             ),
                                                                 html.Div(days_horizon_graphs[i].strftime('%d/%m/%Y'), style={'display': 'none'}, id='pattern_details_date_{}'.format(i)),
                                                             ],
                                                             width=1,
                                                             # style={'position': 'absolute', 'left': '5rem'}
                                                         ),
                                                         dbc.Col(
                                                             dcc.Graph(
                                                                 # figure=draw_pattern_detail_plot(date_max.date()),
                                                                 # figure={},
                                                                 figure=draw_pattern_detail_plot(*get_daily_data(days_horizon_graphs[i]), x_range=get_x_range_for_day(days_horizon_graphs[i])),
                                                                 id='overview_horizon_graph_{}'.format(i),
                                                                 config={
                                                                     'displayModeBar': False
                                                                 },
                                                                 # style={'position': 'absolute', 'left': '100px'}
                                                             )
                                                         ),
                                                         dbc.Col(
                                                             [
                                                                 dbc.Button(children=html.Span([html.I(className="fas fa-caret-down fa-2x")]),
                                                                            color="secondary",
                                                                            className="btn button-icon d-flex align-items-center",
                                                                            size="sm",
                                                                            style={'height': '2rem', 'width': '2rem'},
                                                                            outline=True,
                                                                            id='overview_btn_horizon_expand-{}-'.format(i), n_clicks=0),

                                                                 dbc.Button(children=html.Span([html.I(className="fas fa-eye-slash fa-1x")]),
                                                                            color="secondary",
                                                                            className="btn button-icon d-flex align-items-center",
                                                                            size="sm",
                                                                            style={'height': '2rem', 'width': '2rem', 'position': 'relative', 'left': '2.2rem', 'top': '-2rem'},
                                                                            outline=True,
                                                                            id='btn_agp_visible-{}-'.format(i),
                                                                            n_clicks=0
                                                                            ),
                                                             ],
                                                             width=1
                                                         )
                                                     ],
                                                 ),
                                                 style={'display': 'inline'}
                                             )
                                             for i in range(initial_number_of_days)
                                         ]
                                         + [
                                             html.Div(
                                                 id='pattern_detail_agp_div_{}'.format(i),
                                                 children=
                                                 dbc.Row(
                                                     [
                                                         dbc.Col(
                                                             [
                                                                 html.Button(children='',
                                                                             id='btn_agp_date-{}-'.format(i),
                                                                             n_clicks=0,
                                                                             style=buttons_style_agp_date),
                                                                 html.Div('', style={'display': 'none'}, id='pattern_details_date_{}'.format(i)),
                                                             ],
                                                         ),
                                                         dbc.Col(
                                                             dcc.Graph(
                                                                 # figure=draw_pattern_detail_plot(date_max.date()),
                                                                 figure={},
                                                                 # figure=draw_pattern_detail_plot(*get_daily_data(days_pattern_detail[i]), x_range=get_x_range_for_day(days_pattern_detail[i])),
                                                                 id='overview_horizon_graph_{}'.format(i),
                                                                 config={
                                                                     'displayModeBar': False
                                                                 },
                                                                 # style={'padding': '0% 0% 0% 14px'}
                                                             )
                                                         ),
                                                         dbc.Col(
                                                             [
                                                                 dbc.Button(children=html.Span([html.I(className="fas fa-caret-down fa-2x")]),
                                                                            color="secondary",
                                                                            className="btn button-icon d-flex align-items-center",
                                                                            size="sm",
                                                                            style={'height': '2rem', 'width': '2rem'},
                                                                            outline=True,
                                                                            id='overview_btn_horizon_expand-{}-'.format(i), n_clicks=0),
                                                                 dbc.Button(children=html.Span([html.I(className="fas fa-eye-slash fa-1x")]),
                                                                            color="secondary",
                                                                            className="btn button-icon d-flex align-items-center",
                                                                            size="sm",
                                                                            style={'height': '2rem', 'width': '2rem', 'position': 'relative', 'left': '2.2rem', 'top': '-2rem'},
                                                                            outline=True,
                                                                            id='btn_agp_visible-{}-'.format(i),
                                                                            n_clicks=0
                                                                            ),
                                                             ],
                                                             width=1
                                                         )
                                                     ],
                                                 ),
                                                 style={'display': 'none'}
                                             )
                                             for i in range(initial_number_of_days, num_horizon_graphs)
                                         ]

                                ,
                                style={'display': 'block'},
                                color='white',
                                outline=True,
                                body=True
                            )
                        ],
                        )
                    ],
                    width=8,
                ),
                dbc.Col(
                    html.Div(
                        children=[
                            ################################################################################
                            # STATISTICS AGP
                            ################################################################################
                            html.Div(
                                [
                                    ################################################################################
                                    # BASIC METRICS
                                    ################################################################################
                                    html.Br(),
                                    dbc.Row(
                                        [
                                            dbc.Col(
                                                dbc.Card(
                                                    [
                                                        dbc.Row(html.Div(children='eA1c', className='c-stats-card-tir__title')),
                                                        html.Div(
                                                            [
                                                                html.Div(children=str(round(stats['ea1c'], 1)), id='overview_stats_sgv_ea1c', className='c-stats-card__value'),
                                                                html.Div(children=' %', className='c-stats-card-tir__unit'),
                                                            ],
                                                            style={'text-align': 'center', 'padding': '12% 0'}
                                                        ),
                                                    ],
                                                    style={'height': '10rem', 'width': '10rem'},
                                                    className='image-border',
                                                ),
                                            ),
                                            dbc.Col(
                                                dbc.Card(
                                                    [
                                                        dbc.Row(html.Div(children='Average', className='c-stats-card-tir__title')),
                                                        html.Div(
                                                            [
                                                                html.Div(children=str(int(stats['mean'])), id='overview_stats_sgv_mean', className='c-stats-card__value'),
                                                                html.Div(children=' mg/dL', className='c-stats-card-tir__unit'),
                                                            ],
                                                            style={'text-align': 'center', 'padding': '12% 0'}
                                                        ),
                                                    ],
                                                    style={'height': '10rem', 'width': '10rem'},
                                                    className='image-border',
                                                ),
                                            ),
                                            dbc.Col(
                                                dbc.Card(
                                                    [
                                                        dbc.Row(html.Div(children='CV', className='c-stats-card-tir__title')),
                                                        html.Div(
                                                            [
                                                                html.Div(children=str(int(stats['std'] / stats['mean'] * 100)), id='overview_stats_sgv_std', className='c-stats-card__value'),
                                                                html.Div(children=' %', className='c-stats-card-tir__unit'),
                                                            ],
                                                            style={'text-align': 'center', 'padding': '12% 0'}
                                                        ),
                                                    ],
                                                    style={'height': '10rem', 'width': '10rem'},
                                                    className='image-border',
                                                ),
                                            )
                                        ]
                                    ),
                                    ################################################################################
                                    # TIME IN RANGE
                                    ################################################################################
                                    html.Div(children='Time in Range', style={'font-size': 'small', 'padding': '3% 3%', 'font-weight': 'bold'}),
                                    dbc.Row(
                                        [
                                            dbc.Col(
                                                [
                                                    dbc.Card(
                                                        [
                                                            dbc.Row(html.Div(children='Low', className='c-stats-card-tir__title')),
                                                            dbc.Row(html.Div(children='54 - 70 mg/dl', className='c-stats-card-tir__range')),
                                                            html.Div(
                                                                [
                                                                    html.Div(children=str(tir[1]), id='overview_stats_low', className='c-stats-card-tir__value-x', style={'color': colors['bg_low']}),
                                                                    html.Div(children=' %', className='c-stats-card-tir__unit'),
                                                                ],
                                                                className='c-stats-card-tir__wrapper'
                                                            ),
                                                            dbc.Row(html.Div(children=calculate_tir_time(tir[1]), id='overview_stats_low_time', className='c-stats-card-tir__time',
                                                                             style={'top': '-2rem', 'color': colors['bg_low']})),
                                                        ],
                                                        style={'height': '8rem', 'width': '10rem'},
                                                        className='image-border-top',
                                                    ),
                                                    dbc.Card(
                                                        [
                                                            dbc.Row(html.Div(children='Very Low', className='c-stats-card-tir__title')),
                                                            dbc.Row(html.Div(children='< 54 mg/dl', className='c-stats-card-tir__range')),
                                                            html.Div(
                                                                [
                                                                    html.Div(children=str(tir[0]), id='overview_stats_very_low', className='c-stats-card-tir__value-x', style={'color': colors['bg_very_low']}),
                                                                    html.Div(children=' %', className='c-stats-card-tir__unit'),
                                                                ],
                                                                className='c-stats-card-tir__wrapper'
                                                            ),
                                                            dbc.Row(
                                                                html.Div(children=calculate_tir_time(tir[0]), id='overview_stats_very_low_time', className='c-stats-card-tir__time',
                                                                         style={'top': '-2rem', 'color': colors['bg_very_low']})),
                                                        ],
                                                        style={'height': '8rem', 'width': '10rem'},
                                                        className='image-border-bottom',
                                                    )
                                                ]
                                            ),
                                            dbc.Col(
                                                dbc.Card(
                                                    [
                                                        dbc.Row(html.Div(children='Target', className='c-stats-card-tir__title')),
                                                        dbc.Row(html.Div(children='70 - 180 mg/dl', className='c-stats-card-tir__range')),
                                                        html.Div(
                                                            [
                                                                html.Div(children=str(tir[2]), id='overview_stats_target', className='c-stats-card-tir__value-xx', style={'color': colors['bg_target']}),
                                                                html.Div(children=' %', className='c-stats-card-tir__unit'),
                                                            ],
                                                            style={'text-align': 'center', 'padding': '15% 0'}
                                                        ),
                                                        dbc.Row(html.Div(children=calculate_tir_time(tir[2]), id='overview_stats_target_time', className='c-stats-card-tir__time', style={'color': colors['bg_target'], })),
                                                    ],
                                                    style={'height': '16rem', 'width': '10rem'},
                                                    className='image-border',
                                                )
                                            ),
                                            dbc.Col(
                                                [
                                                    dbc.Card(
                                                        [
                                                            dbc.Row(html.Div(children='High', className='c-stats-card-tir__title')),
                                                            dbc.Row(html.Div(children='180 - 250 mg/dl', className='c-stats-card-tir__range')),
                                                            html.Div(
                                                                [
                                                                    html.Div(children=str(tir[3]), id='overview_stats_high', className='c-stats-card-tir__value-x', style={'color': colors['bg_high']}),
                                                                    html.Div(children=' %', className='c-stats-card-tir__unit'),
                                                                ],
                                                                className='c-stats-card-tir__wrapper'
                                                            ),
                                                            dbc.Row(html.Div(children=calculate_tir_time(tir[3]), id='overview_stats_high_time', className='c-stats-card-tir__time',
                                                                             style={'text-align': 'center', 'top': '-2rem', 'color': colors['bg_high']})),
                                                        ],
                                                        style={'height': '8rem', 'width': '10rem'},
                                                        className='image-border-top',
                                                    ),
                                                    dbc.Card(
                                                        [
                                                            html.Div(children='Very High', className='c-stats-card-tir__title'),
                                                            html.Div(children='> 250 mg/dl', className='c-stats-card-tir__range'),
                                                            html.Div(
                                                                [
                                                                    html.Div(children=str(tir[4]), id='overview_stats_very_high', className='c-stats-card-tir__value-x', style={'color': colors['bg_very_high']}),
                                                                    html.Div(children=' %', className='c-stats-card-tir__unit'),
                                                                ],
                                                                className='c-stats-card-tir__wrapper'
                                                            ),
                                                            dbc.Row(html.Div(children=calculate_tir_time(tir[4]), id='overview_stats_very_high_time', className="c-stats-card-tir__time",
                                                                             style={'top': '-2rem', 'color': colors['bg_very_high']})),
                                                        ],
                                                        style={'height': '8rem', 'width': '10rem'},
                                                        className='image-border-bottom',
                                                    )
                                                ]
                                            ),
                                        ]
                                    ),
                                    html.Div(
                                        dcc.Graph(
                                            figure=get_tir_plot(tir),
                                            id='overview_stats_tir_graph',
                                            style={'padding': '0rem 0rem 0rem 0.5rem'},
                                            config={
                                                'displayModeBar': False
                                            }
                                        ),
                                        style={'border-radius': '25px 25px 25px 25px', 'overflow': 'hidden'}
                                    ),

                                    ################################################################################
                                    # TREATMENTS
                                    ################################################################################
                                    html.Div(children='Treatments', style={'font-size': 'small', 'padding': '0% 3% 3%', 'font-weight': 'bold'}),
                                    dbc.Row(
                                        [
                                            dbc.Col(
                                                dbc.Card(
                                                    [
                                                        dbc.Row(html.Div(children='Carbs', className='c-stats-card-tir__title')),
                                                        html.Div(
                                                            [
                                                                html.Div(children=carbs_sum, id='overview_stats_carbs', className='c-stats-card__value', style={'color': colors['carbs']}),
                                                                html.Div(children=' g', className='c-stats-card-tir__unit'),
                                                            ],
                                                            style={'text-align': 'center', 'padding': '12% 0'}
                                                        ),
                                                    ],
                                                    style={'height': '10rem', 'width': '10rem'},
                                                    className='image-border',
                                                ),
                                            ),
                                            dbc.Col(
                                                dbc.Card(
                                                    [
                                                        dbc.Row(html.Div(children='Insulin ({} U)'.format(bolus_sum + basal_sum), className='c-stats-card-tir__title')),
                                                        html.Div(
                                                            dbc.Row(
                                                                [
                                                                    dbc.Col(
                                                                        [
                                                                            html.Div(
                                                                                [
                                                                                    html.Div(children=bolus_sum, id='overview_stats_bolus', className='c-stats-card__value', style={'color': colors['bolus'], }),
                                                                                    html.Div(children=' U', className='c-stats-card-tir__unit'),
                                                                                ],
                                                                                style={'text-align': 'center', 'padding': '12% 0'}
                                                                            ),
                                                                            dbc.Row(html.Div(children=str(round(bolus_sum / (bolus_sum + basal_sum) * 100)) + ' %',
                                                                                             className='c-stats-card-treat__prctg',
                                                                                             style={'top': '-2rem', 'color': colors['bolus']})),
                                                                        ],
                                                                        style={'border-right': '1px solid', 'border-color': 'LightGray'}
                                                                    ),
                                                                    dbc.Col(
                                                                        [
                                                                            html.Div(
                                                                                [
                                                                                    html.Div(children=basal_sum, id='overview_stats_basal', className='c-stats-card__value', style={'color': colors['basal'], }),
                                                                                    html.Div(children=' U', className='c-stats-card-tir__unit'),
                                                                                ],
                                                                                style={'text-align': 'center', 'padding': '12% 0'}
                                                                            ),
                                                                            dbc.Row(
                                                                                html.Div(children=str(round(basal_sum / (bolus_sum + basal_sum) * 100)) + ' %',
                                                                                         className='c-stats-card-treat__prctg',
                                                                                         style={'top': '-2rem', 'color': colors['basal']})),
                                                                        ]
                                                                    ),
                                                                ],
                                                                className="h-75",
                                                            ),
                                                        ),

                                                    ],
                                                    style={'height': '10rem', 'width': '22rem'},
                                                    className='image-border',
                                                ),
                                                width=8
                                            )
                                        ]
                                    )
                                ]
                            )
                        ],
                        style={'position': 'fixed', 'top': '0', 'z-index': '900', 'height': '100%', "background-color": "#f8f9fa"}
                    ),
                    width=4,
                ),
            ]
        )
    ]
)

# layout_agp = html.Div(
#                     [
#                         html.Div(
#                             dbc.Card(
#                                 children=
#                                 [
#                                     # dbc.CardHeader('AGP'),
#                                     dbc.CardBody(
#                                         dcc.Graph(
#                                             id='seasonal_day_graph',
#                                             figure=draw_seasonal_graph_day(start_date, end_date),
#                                             config={
#                                                 'displayModeBar': False
#                                             }
#                                         )
#                                     )
#                                 ],
#                                 style={'padding': '0rem 0rem 0rem 60px'},  # ice: 6rem
#                                 className='image-border',
#                                 # color=colors['background']
#                             ),
#                             style={'position': 'sticky', 'top': '0', 'z-index': '900'},
#                         ),
#                         html.Div(
#                             children=dcc.Graph(
#                                 figure=agp_xaxis()
#                             ),
#                             style={'position': 'fixed', 'bottom': '0', 'z-index': '9000', 'padding': '0rem 0rem 0rem 60px'}),
#                         # pattern_details_agp_card
#                     ]
# )
