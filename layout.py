from datetime import timedelta, datetime, date
import json
import dash
import dash_bootstrap_components as dbc
from dash import dcc, html, ctx
from dash_extensions.enrich import Output, DashProxy, Input, MultiplexerTransform, State, callback_context, ALL
from aggregations import draw_seasonal_graph_day, draw_seasonal_graph, draw_agp_carbs_bolus, draw_full_agp, agp_xaxis
from colors import colors_patterns, colors_heatmap, targets_heatmap, get_prebolus_button_color, get_button_text_color
from daily import draw_daily_plot
import numpy as np
from insights import draw_pattern_overview, add_time_before_after, get_insight_dataset, get_insight_clusters, draw_hierarchical_pattern_overview, get_logs_meals, get_logs_from_indices, get_insight_data_meals, \
    get_time_of_day_from_number, get_curve_overview_plot, get_dataset, filter_function_time_of_day, filter_function_meal_size, get_insight_data_hypos
from overview import draw_horizon_graph, get_daily_data, get_x_range_for_day, get_non_periodic_data, draw_overview_daily_curve_detailed
from preprocessing import dates, logs_sgv, date_max, date_min, start_date, end_date, sgv_array_for_agp, date_dict, logs_carbs, logs_insulin, logs_br_default, start_date_insights
from statistics import get_tir_plot, get_statistics_day, get_statistics_days
from helpers import convert_datestring, get_df_between_dates, get_tir, get_statistics, check_timebox, get_log_indices, calculate_tir_time, get_mean_per_day, get_df_of_date
from variables import num_horizon_graphs, n_filters, initial_number_of_days, num_insight_details, time_before_meal, time_after_meal, num_insight_patterns
from assets.styles import *
import re
from datetime import datetime, timedelta
import warnings
from pandas.errors import SettingWithCopyWarning

warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)

# Connect to main app.py file
FONT_AWESOME = "https://use.fontawesome.com/releases/v5.7.2/css/all.css"
app = DashProxy(prevent_initial_callbacks=True, transforms=[MultiplexerTransform()], suppress_callback_exceptions=True, external_stylesheets=[dbc.themes.BOOTSTRAP, FONT_AWESOME])
app.title = 'Marjorie'

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
                    html.Div(html.Img(src=app.get_asset_url('marjorie-logo.svg'), style={'width': '70%'}), style={'text-align': 'center'}),
                    html.Div(html.Img(src=app.get_asset_url('marjorie-text.svg'), style={'width': '90%', 'margin-top': '1rem', 'margin-bottom': '2rem'}), style={'text-align': 'center'}),
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
                            id='daily_graph',
                            config={
                                'displayModeBar': False
                            }
                        ),
                    )
                ],
                className='image-border',
                style={'padding': '0 2em'}
            ),
            width=8,
            md=8,
            xs=12
        ),
        dbc.Col(
            html.Div(
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
                            display_format='YYYY/MM/DD',
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
                                                    dbc.Row(
                                                        html.Div(children=calculate_tir_time(tir[1]), id='stats_daily_low_time', className='c-stats-card-tir__time', style={'top': '-2rem', 'color': colors['bg_low']})),
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
                                                        html.Div(children=calculate_tir_time(tir[0]), id='stats_daily_very_low_time', className='c-stats-card-tir__time',
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
                style=SIDEBAR_STYLE
            ),
            width=4, md=4, className="sm-hide"
        )
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
                                                display_format='YYYY/MM/DD',
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
                                                                figure=draw_seasonal_graph('week', start_date, end_date),
                                                                config={
                                                                    'displayModeBar': False
                                                                }
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
                        style=SIDEBAR_STYLE
                    ),
                    width=4,
                ),
            ]
        )
    ]
)

def create_horizon_graph(id, day):
    horizon_graph = html.Div(
        id={"index": id, "type": "horizon_card"},
        children=
        dbc.Row(
            [
                dbc.Col(
                    [
                        html.Button(children=day.strftime('%a, %m/%d'),
                                    id={"index": id, "type": "horizon_date_displayed"},
                                    n_clicks=0,
                                    style=buttons_style_agp_date
                                    ),
                        html.Div(day.strftime('%d/%m/%Y'), style={'display': 'none'}, id={"index": id, "type": "horizon_date_info"}, ),
                    ],
                    width=1,
                ),
                dbc.Col(
                    dcc.Graph(
                        # figure={},
                        figure=draw_horizon_graph(*get_daily_data(day), x_range=get_x_range_for_day(day)),
                        id={"index": id, "type": "horizon_graph"},
                        config={
                            'displayModeBar': False
                        },
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
                                   id={"index": id, "type": "btn_horizon_graph_expand"}, n_clicks=0),

                        dbc.Button(children=html.Span([html.I(className="fas fa-eye-slash fa-1x")]),
                                   color="secondary",
                                   className="btn button-icon d-flex align-items-center",
                                   size="sm",
                                   style={'height': '2rem', 'width': '2rem', 'position': 'relative', 'left': '2.2rem', 'top': '-2rem'},
                                   outline=True,
                                   id={"index": id, "type": "btn_horizon_graph_hide"},
                                   n_clicks=0
                                   ),
                    ],
                    width=1
                )
            ],
        ),
        style={'display': 'inline'}
    )
    return horizon_graph

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
                                                display_format='YYYY/MM/DD',
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
                                                                figure=draw_seasonal_graph('week', start_date, end_date),
                                                                config={
                                                                    'displayModeBar': False
                                                                }
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
                        ],
                        ),
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
                                    figure=agp_xaxis(),
                                    config={
                                        'displayModeBar': False
                                    }
                                ),
                                style={'position': 'fixed', 'bottom': '0', 'z-index': '9000', 'padding': '0rem 0rem 0rem 60px'}
                            ),
                            ################################################################################
                            # HORIZON GRAPHS
                            ################################################################################
                            dbc.Card(
                                id='overview_horizon_graphs',
                                children=[create_horizon_graph(id, date) for id, date in enumerate(days_horizon_graphs)],
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
                            # STATISTICS OVERVIEW
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
                        style=SIDEBAR_STYLE
                    ),
                    width=4,
                ),
            ]
        )
    ]
)

################################################################################
# INSIGHTS SECTION
################################################################################

meals_n_clusters_, meals_bar_graphs, meals_graph_all_curves, meals_graphs_insights, meals_start_bgs, meals_time_between, meals_carbs_sums, meals_end_bgs, meals_bolus_sums = get_insight_data_meals()
meals_styles = [{'display': 'inline'}] * meals_n_clusters_ + [{'display': 'none'}] * (num_insight_patterns - meals_n_clusters_)
hypos_n_clusters_, hypos_bar_graphs, hypos_graph_all_curves, hypos_graphs_insights, hypos_start_bgs, hypos_end_bgs, hypos_carb_avg_before, hypos_carb_avg_after, hypos_bolus_avg_before, \
hypos_bolus_avg_after = get_insight_data_hypos()
hypos_styles = [{'display': 'inline'}] * hypos_n_clusters_ + [{'display': 'none'}] * (num_insight_patterns - hypos_n_clusters_)

layout_insights = dbc.Tabs(
    [dbc.Tab(
        ################################################################################
        # MEAL INSIGHTS
        ################################################################################
        dbc.Row(
            [
                ################################################################################
                # MEAL INSIGHTS: PATTERN SCREEN
                ################################################################################
                dbc.Col(
                    ################################################################################
                    # MEAL INSIGHTS: DISPLAYED PATTERN DIVS
                    ################################################################################
                    [
                        # html.Div(children='{} patterns were found.'.format(n_clusters_), id='n_patterns_meals', style={'font-family': font, 'font-size': 'medium', 'padding': '1% 0% 2% 2%'}),
                        html.Div(
                            [
                                ################################################################################
                                # MEAL INSIGHTS: PATTERN HEADING
                                ################################################################################
                                dbc.Row(
                                    [
                                        dbc.Col(
                                            html.Div('Pattern {}'.format(i + 1), style={'font-family': font, 'font-size': 'small', 'padding': '10% 0% 0% 10%', 'font-weight': 'bold'}),
                                            width=2
                                        ),
                                    ]
                                ),
                                ################################################################################
                                # MEAL INSIGHTS: PATTERN CONTENT
                                ################################################################################
                                dbc.Row(
                                    [
                                        ################################################################################
                                        # MEAL INSIGHTS: PATTERN GRAPH
                                        ################################################################################
                                        dbc.Col(
                                            [
                                                dcc.Graph(
                                                    figure=meals_graphs_insights[i],
                                                    id='insights_meals_overview_graph_{}'.format(i),
                                                    config={
                                                        'displayModeBar': False
                                                    }
                                                ),
                                            ],
                                        ),
                                        ################################################################################
                                        # MEAL INSIGHTS: PATTERN STATISTICS
                                        ################################################################################
                                        dbc.Col(
                                            [
                                                ################################################################################
                                                # MEAL INSIGHTS: BAR CHART
                                                ################################################################################
                                                dbc.Row(
                                                    dcc.Graph(
                                                        figure=meals_bar_graphs[i],
                                                        id='insights_meals_bar_graph_{}'.format(i),
                                                        config={
                                                            'displayModeBar': False
                                                        }
                                                    ),
                                                    style={'padding': '0 0 5%'}
                                                ),
                                                ################################################################################
                                                # MEAL INSIGHTS: STATISTIC CARDS
                                                ################################################################################
                                                dbc.Row(
                                                    [
                                                        dbc.Card(
                                                            [
                                                                dbc.Row(html.Div(children='before',
                                                                                 style={'font-family': font, 'font-size': 'xx-small', 'padding': '10% 0% 0%', 'text-align': 'center', 'font-weight': 'bold'})),
                                                                html.Div(
                                                                    [
                                                                        html.Div(children=str(meals_start_bgs[i]), id={'type': 'insights_meals_sgv_before', 'index': i},
                                                                                 style={'font-family': font, 'font-size': 'small', 'font-weight': 'bold',
                                                                                        'display': 'inline-block'}),
                                                                        html.Div(children='  mg/dL', style={'font-family': font, 'font-size': '20%', 'display': 'inline-block'}),
                                                                    ],
                                                                    style={'text-align': 'center', 'padding': '12% 0'}
                                                                ),
                                                            ],
                                                            style={'height': '7rem', 'width': '7rem', 'margin': '0 2% 0 0', 'color': get_button_text_color(meals_start_bgs[i] / 380)},
                                                            className='image-border-top',
                                                            color=colors_heatmap[list(np.array(targets_heatmap) > meals_start_bgs[i]).index(True) - 1],
                                                            id={'type': 'insights_meals_card_sgv_before', 'index': i}
                                                        ),
                                                        dbc.Card(
                                                            [
                                                                dbc.Row(html.Div(children='prebolus',
                                                                                 style={'font-family': font, 'font-size': 'xx-small', 'padding': '3% 0% 0%', 'text-align': 'center', 'font-weight': 'bold'})),
                                                                html.Div(
                                                                    [
                                                                        html.Div(children=str(meals_time_between[i]), id={'type': 'insights_meals_interval', 'index': i},
                                                                                 style={'font-family': font, 'font-size': 'small', 'font-weight': 'bold',
                                                                                        'display': 'inline-block'}),
                                                                        html.Div(children='  min', style={'font-family': font, 'font-size': '20%', 'display': 'inline-block'}),
                                                                    ],
                                                                    style={'text-align': 'center', 'padding': '0% 0'}
                                                                ),
                                                            ],
                                                            style={'height': '4rem', 'width': '7rem', 'margin': '0 0 0 0', 'color': get_button_text_color(meals_time_between[i] / 30)},
                                                            className='image-border-left',
                                                            color=get_prebolus_button_color(meals_time_between[i]),
                                                            id={'type': 'insights_meals_card_interval', 'index': i}
                                                        ),
                                                        dbc.Card(
                                                            [
                                                                dbc.Row(html.Div(children='factor',
                                                                                 style={'font-family': font, 'font-size': 'xx-small', 'padding': '3% 0% 0%', 'text-align': 'center', 'font-weight': 'bold'})),
                                                                html.Div(
                                                                    [
                                                                        html.Div(children=str(round(meals_carbs_sums[i] / meals_bolus_sums[i])), id={'type': 'insights_meals_factor', 'index': i},
                                                                                 style={'font-family': font, 'font-size': 'small',
                                                                                        'font-weight': 'bold',
                                                                                        'display': 'inline-block', }),
                                                                        html.Div(children='  g/U', style={'font-family': font, 'font-size': '20%', 'display': 'inline-block'}),
                                                                    ],
                                                                    style={'text-align': 'center', 'padding': '0% 0'}
                                                                ),
                                                            ],
                                                            style={'height': '4rem', 'width': '7rem', 'margin': '0 0 0 0'},
                                                            className='image-border-right',
                                                            color='rgba(157, 164, 169,' + str(min((meals_carbs_sums[i] / meals_bolus_sums[i]) / 10, 1)) + ')',
                                                            id={'type': 'insights_meals_card_factor', 'index': i}
                                                        ),
                                                    ],
                                                ),
                                                dbc.Row(
                                                    [
                                                        dbc.Card(
                                                            [
                                                                dbc.Row(html.Div(children='after',
                                                                                 style={'font-family': font, 'font-size': 'xx-small', 'padding': '10% 0% 0%', 'text-align': 'center', 'font-weight': 'bold'})),
                                                                html.Div(
                                                                    [
                                                                        html.Div(children=str(meals_end_bgs[i]), id={'type': 'insights_meals_sgv_after', 'index': i},
                                                                                 style={'font-family': font, 'font-size': 'small', 'font-weight': 'bold',
                                                                                        'display': 'inline-block'}),
                                                                        html.Div(children='  mg/dL', style={'font-family': font, 'font-size': '20%', 'display': 'inline-block'}),
                                                                    ],
                                                                    style={'text-align': 'center', 'padding': '12% 0'}
                                                                ),
                                                            ],
                                                            style={'height': '7rem', 'width': '7rem', 'margin': '0 2% 0 0', 'color': get_button_text_color(meals_end_bgs[i] / 380)},
                                                            color=colors_heatmap[list(np.array(targets_heatmap) > meals_end_bgs[i]).index(True) - 1],
                                                            className='image-border-bottom',
                                                            id={'type': 'insights_meals_card_sgv_after', 'index': i}
                                                        ),
                                                        dbc.Card(
                                                            [
                                                                dbc.Row(html.Div(children='meal size',
                                                                                 style={'font-family': font, 'font-size': 'xx-small', 'padding': '20% 0% 0%', 'text-align': 'center', 'font-weight': 'bold'})),
                                                                html.Div(
                                                                    [
                                                                        html.Div(children=str(round(meals_carbs_sums[i])), id={'type': 'insights_meals_meal_size', 'index': i},
                                                                                 style={'font-family': font, 'font-size': 'small', 'font-weight': 'bold',
                                                                                        'display': 'inline-block', }),
                                                                        html.Div(children='  g', style={'font-family': font, 'font-size': '20%', 'display': 'inline-block'}),
                                                                    ],
                                                                    style={'text-align': 'center', 'padding': '12% 0'}
                                                                ),
                                                            ],
                                                            style={'height': '9rem', 'width': '7rem', 'position': 'relative', 'top': '-2rem'},
                                                            className='image-border',
                                                            color=colors['carbs'][:-2] + str(min((meals_carbs_sums[i] - 20) / 90, 1)) + ')',
                                                            id={'type': 'insights_meals_card_meal_size', 'index': i}
                                                        ),

                                                        dbc.Card(
                                                            [
                                                                dbc.Row(html.Div(children='bolus',
                                                                                 style={'font-family': font, 'font-size': 'xx-small', 'padding': '20% 0% 0%', 'text-align': 'center', 'font-weight': 'bold'})),
                                                                html.Div(
                                                                    [
                                                                        html.Div(children=str(round(meals_bolus_sums[i], 1)), id={'type': 'insights_meals_bolus', 'index': i},
                                                                                 style={'font-family': font, 'font-size': 'small', 'font-weight': 'bold', 'display': 'inline-block', }),
                                                                        html.Div(children='  U', style={'font-family': font, 'font-size': '20%', 'display': 'inline-block'}),
                                                                    ],
                                                                    style={'text-align': 'center', 'padding': '12% 0'}
                                                                ),
                                                            ],
                                                            style={'height': '9rem', 'width': '7rem', 'position': 'relative', 'top': '-2rem'},
                                                            className='image-border',
                                                            color=colors['bolus'][:-2] + str(min((meals_bolus_sums[i] - 5) / 14, 1)) + ')',
                                                            id={'type': 'insights_meals_card_bolus', 'index': i}
                                                        ),
                                                    ],
                                                ),
                                            ],
                                            width=4
                                        )
                                    ],
                                    style={'padding': '0%', 'margin': '0%'}
                                ),
                            ],
                            style=meals_styles[i],
                            id={'type': 'insights_meals_pattern_card', 'index': i}
                        )
                        for i in range(0, num_insight_patterns)
                    ], width=9),

                ################################################################################
                # MEAL INSIGHTS: FILTER SIDEBAR
                ################################################################################
                dbc.Col(
                    html.Div(
                        [
                            ################################################################################
                            # MEAL INSIGHTS: SIDEBAR TITLE
                            ################################################################################
                            html.Div('FILTER', style={'font-family': font, 'font-size': 'small', 'padding': '0% 0% 0% 0%', 'font-weight': 'bold'}),

                            ################################################################################
                            # MEAL INSIGHTS: PATTERN GRAPHS OVERLAYED
                            ################################################################################
                            dbc.Row(
                                dcc.Graph(
                                    figure=meals_graph_all_curves,
                                    id='insights_meals_graph_all_curves',
                                    config={
                                        'displayModeBar': False
                                    }
                                ),
                            ),
                            html.Div(style={'padding': '0% 0 10%'}),

                            ################################################################################
                            # MEAL INSIGHTS: FILTER CHECKLIST TIME OF DAY
                            ################################################################################
                            html.Div(
                                [
                                    dbc.Label("Time of day", style={'font-weight': 'bold'}),
                                    dbc.Checklist(
                                        options=[
                                            {"label": "Morning (6:00 - 11:00)", "value": 1},
                                            {"label": "Noon (11:00 - 16:00)", "value": 2},
                                            {"label": "Evening (16:00 - 24:00)", "value": 3},
                                            {"label": "Night (0:00 - 6:00)", "value": 4},
                                        ],
                                        value=[1, 2, 3, 4],
                                        input_checked_style={
                                            "backgroundColor": "#6c6c6c",
                                            "borderColor": "#6c6c6c",
                                        },
                                        id="insights_meals_checklist_time_of_day",
                                    ),
                                ]
                            ),

                            ################################################################################
                            # MEAL INSIGHTS: FILTER SLIDER MEAL SIZE
                            ################################################################################
                            dbc.Label("Meal size", style={'padding': '10% 0 0', 'font-weight': 'bold'}, html_for="range-slider-meal-size"),
                            dcc.RangeSlider(id="insights_meals_range_slider_meal_size",
                                            min=40,
                                            max=150,
                                            step=10,
                                            marks={
                                                0: '0 g',
                                                25: '25 g',
                                                50: '50 g',
                                                75: '75 g',
                                                100: '100 g',
                                                125: '125 g',
                                                150: '150 g',
                                            },
                                            value=[0, 150],
                                            tooltip={"placement": "bottom", "always_visible": True},
                                            ),
                            html.Div(style={'padding': '0% 0 10%'}),

                            ################################################################################
                            # MEAL INSIGHTS: APPLY BUTTON
                            ################################################################################
                            dbc.Button("Apply",
                                       id='insights_meals_filter_apply_btn',
                                       color="secondary",
                                       className="me-1",
                                       n_clicks=0,
                                       disabled=True,
                                       ),
                        ],
                        style=SIDEBAR_STYLE,
                    ), width=3)
            ]
        ), label="Meals", labelClassName='text-dark', activeTabClassName="fw-bold"
    ),
        dbc.Tab(
            ################################################################################
            # HYPO INSIGHTS
            ################################################################################
            dbc.Row(
                [
                    ################################################################################
                    # HYPO INSIGHTS: PATTERN SCREEN
                    ################################################################################
                    dbc.Col(
                        ################################################################################
                        # HYPO INSIGHTS: DISPLAYED PATTERN DIVS
                        ################################################################################
                        [
                            # html.Div(children='{} patterns were found.'.format(n_clusters_), id='n_patterns_meals', style={'font-family': font, 'font-size': 'medium', 'padding': '1% 0% 2% 2%'}),
                            html.Div(
                                [
                                    ################################################################################
                                    # HYPO INSIGHTS: PATTERN HEADING
                                    ################################################################################
                                    dbc.Row(
                                        [
                                            dbc.Col(
                                                html.Div('Pattern {}'.format(i + 1), style={'font-family': font, 'font-size': 'small', 'padding': '10% 0% 0% 10%', 'font-weight': 'bold'}),
                                                width=2
                                            ),
                                        ]
                                    ),
                                    ################################################################################
                                    # HYPO INSIGHTS: PATTERN CONTENT
                                    ################################################################################
                                    dbc.Row(
                                        [
                                            ################################################################################
                                            # HYPO INSIGHTS: PATTERN GRAPH
                                            ################################################################################
                                            dbc.Col(
                                                [
                                                    dcc.Graph(
                                                        figure=hypos_graphs_insights[i],
                                                        id='insights_hypos_overview_graph_{}'.format(i),
                                                        config={
                                                            'displayModeBar': False
                                                        }
                                                    ),
                                                ],
                                            ),
                                            ################################################################################
                                            # HYPO INSIGHTS: PATTERN STATISTICS
                                            ################################################################################
                                            dbc.Col(
                                                [
                                                    ################################################################################
                                                    # HYPO INSIGHTS: BAR CHART
                                                    ################################################################################
                                                    dbc.Row(
                                                        dcc.Graph(
                                                            figure=hypos_bar_graphs[i],
                                                            id='insights_hypos_bar_graph_{}'.format(i),
                                                            config={
                                                                'displayModeBar': False
                                                            }
                                                        ),
                                                        style={'padding': '0 0 5%'}
                                                    ),
                                                    # dbc.Row(html.Div('Mostly occurring at {}'.format(most_occurring[i]))),
                                                    ################################################################################
                                                    # HYPO INSIGHTS: STATISTIC CARDS
                                                    ################################################################################
                                                    dbc.Row(
                                                        [
                                                            dbc.Card(
                                                                [
                                                                    dbc.Row(html.Div(children='before hypo',
                                                                                     style={'font-family': font, 'font-size': 'xx-small', 'padding': '10% 0% 0%', 'text-align': 'center', 'font-weight': 'bold'})),
                                                                    html.Div(
                                                                        [
                                                                            html.Div(children=str(hypos_start_bgs[i]), id={'type': 'insights_hypos_sgv_before', 'index': i},
                                                                                     style={'font-family': font, 'font-size': 'small', 'font-weight': 'bold',
                                                                                            'display': 'inline-block'}),
                                                                            html.Div(children='  mg/dL', style={'font-family': font, 'font-size': '20%', 'display': 'inline-block'}),
                                                                        ],
                                                                        style={'text-align': 'center', 'padding': '12% 0'}
                                                                    ),
                                                                ],
                                                                style={'height': '7rem', 'width': '7rem', 'margin': '0 2% 0 0', 'color': get_button_text_color(hypos_start_bgs[i] / 380)},
                                                                className='image-border-top',
                                                                color=colors_heatmap[list(np.array(targets_heatmap) > hypos_start_bgs[i]).index(True) - 1],
                                                                id={'type': 'insights_hypos_card_sgv_before', 'index': i}
                                                            ),
                                                            dbc.Card(
                                                                [
                                                                    dbc.Row(html.Div(children='before hypo',
                                                                                     style={'font-family': font, 'font-size': 'xx-small', 'padding': '10% 0% 0%', 'text-align': 'center', 'font-weight': 'bold'})),
                                                                    dbc.Row(html.Div(children='meal size',
                                                                                     style={'font-family': font, 'font-size': 'xx-small', 'padding': '0% 0% 0%', 'text-align': 'center', 'font-weight': 'bold'})),
                                                                    html.Div(
                                                                        [
                                                                            html.Div(children=str(round(hypos_carb_avg_before[i])), id={'type': 'insights_hypos_carb_avg_before', 'index': i},
                                                                                     style={'font-family': font, 'font-size': 'small', 'font-weight': 'bold',
                                                                                            'display': 'inline-block', }),
                                                                            html.Div(children='  g', style={'font-family': font, 'font-size': '20%', 'display': 'inline-block'}),
                                                                        ],
                                                                        style={'text-align': 'center', 'padding': '0% 0'}
                                                                    ),
                                                                ],
                                                                style={'height': '7rem', 'width': '7rem', 'margin': '0 2% 0 0', 'position': 'relative', 'top': '0rem'},
                                                                className='image-border-top',
                                                                color=colors['carbs'][:-2] + str(min((hypos_carb_avg_before[i]) / 60, 1)) + ')',
                                                                id={'type': 'insights_hypos_card_carb_avg_before', 'index': i}
                                                            ),
                                                            dbc.Card(
                                                                [
                                                                    dbc.Row(html.Div(children='before hypo',
                                                                                     style={'font-family': font, 'font-size': 'xx-small', 'padding': '10% 0% 0%', 'text-align': 'center', 'font-weight': 'bold'})),
                                                                    dbc.Row(html.Div(children='bolus',
                                                                                     style={'font-family': font, 'font-size': 'xx-small', 'padding': '0% 0% 0%', 'text-align': 'center', 'font-weight': 'bold'})),
                                                                    html.Div(
                                                                        [
                                                                            html.Div(children=str(round(hypos_bolus_avg_before[i], 1)), id={'type': 'insights_hypos_bolus_avg_before', 'index': i}, style={'font-family':
                                                                                                                                                                                                               font,
                                                                                                                                                                                                           'font-size': 'small',
                                                                                                                                                                                                           'font-weight': 'bold',
                                                                                                                                                                                                           'display': 'inline-block', }),
                                                                            html.Div(children='  U', style={'font-family': font, 'font-size': '20%', 'display': 'inline-block'}),
                                                                        ],
                                                                        style={'text-align': 'center', 'padding': '0% 0'}
                                                                    ),
                                                                ],
                                                                style={'height': '7rem', 'width': '7rem', 'margin': '0 0 0 0', 'position': 'relative', 'top': '0rem'},
                                                                className='image-border-top',
                                                                color=colors['bolus'][:-2] + str(min((hypos_bolus_avg_before[i]) / 5, 1)) + ')',
                                                                id={'type': 'insights_hypos_card_bolus_avg_before', 'index': i}
                                                            ),
                                                        ],
                                                    ),
                                                    dbc.Row(
                                                        [
                                                            dbc.Card(
                                                                [
                                                                    dbc.Row(html.Div(children='after hypo',
                                                                                     style={'font-family': font, 'font-size': 'xx-small', 'padding': '10% 0% 0%', 'text-align': 'center', 'font-weight': 'bold'})),
                                                                    html.Div(
                                                                        [
                                                                            html.Div(children=str(hypos_end_bgs[i]), id={'type': 'insights_hypos_sgv_after', 'index': i},
                                                                                     style={'font-family': font, 'font-size': 'small', 'font-weight': 'bold',
                                                                                            'display': 'inline-block', }),
                                                                            html.Div(children='  mg/dL', style={'font-family': font, 'font-size': '20%', 'display': 'inline-block'}),
                                                                        ],
                                                                        style={'text-align': 'center', 'padding': '12% 0'}
                                                                    ),
                                                                ],
                                                                style={'height': '7rem', 'width': '7rem', 'margin': '0 2% 0 0', 'color': get_button_text_color(hypos_end_bgs[i] / 380)},
                                                                color=colors_heatmap[list(np.array(targets_heatmap) > hypos_end_bgs[i]).index(True) - 1],
                                                                className='image-border-bottom',
                                                                id={'type': 'insights_hypos_card_sgv_after', 'index': i}
                                                            ),
                                                            dbc.Card(
                                                                [
                                                                    dbc.Row(html.Div(children='after hypo',
                                                                                     style={'font-family': font, 'font-size': 'xx-small', 'padding': '10% 0% 0%', 'text-align': 'center', 'font-weight': 'bold'})),
                                                                    dbc.Row(html.Div(children='meal size',
                                                                                     style={'font-family': font, 'font-size': 'xx-small', 'padding': '0% 0% 0%', 'text-align': 'center', 'font-weight': 'bold'})),
                                                                    html.Div(
                                                                        [
                                                                            html.Div(children=str(round(hypos_carb_avg_after[i])), id={'type': 'insights_hypos_carb_avg_after', 'index': i},
                                                                                     style={'font-family': font, 'font-size': 'small', 'font-weight': 'bold',
                                                                                            'display': 'inline-block', }),
                                                                            html.Div(children='  g', style={'font-family': font, 'font-size': '20%', 'display': 'inline-block'}),
                                                                        ],
                                                                        style={'text-align': 'center', 'padding': '12% 0'}
                                                                    ),
                                                                ],
                                                                style={'height': '7rem', 'width': '7rem', 'position': 'relative', 'top': '0rem', 'margin': '0 2% 0 0'},
                                                                className='image-border-bottom',
                                                                color=colors['carbs'][:-2] + str(min((hypos_carb_avg_after[i]) / 60, 1)) + ')',
                                                                id={'type': 'insights_hypos_card_carb_avg_after', 'index': i}
                                                            ),

                                                            dbc.Card(
                                                                [
                                                                    dbc.Row(html.Div(children='after hypo',
                                                                                     style={'font-family': font, 'font-size': 'xx-small', 'padding': '10% 0% 0%', 'text-align': 'center', 'font-weight': 'bold'})),
                                                                    dbc.Row(html.Div(children='bolus',
                                                                                     style={'font-family': font, 'font-size': 'xx-small', 'padding': '0% 0% 0%', 'text-align': 'center', 'font-weight': 'bold'})),
                                                                    html.Div(
                                                                        [
                                                                            html.Div(children=str(round(hypos_bolus_avg_after[i], 1)), id={'type': 'insights_hypos_bolus_avg_after', 'index': i},
                                                                                     style={'font-family': font, 'font-size': 'small', 'font-weight': 'bold', 'display': 'inline-block', }),
                                                                            html.Div(children='  U', style={'font-family': font, 'font-size': '20%', 'display': 'inline-block'}),
                                                                        ],
                                                                        style={'text-align': 'center', 'padding': '12% 0'}
                                                                    ),
                                                                ],
                                                                style={'height': '7rem', 'width': '7rem', 'position': 'relative', 'top': '0rem'},
                                                                className='image-border-bottom',
                                                                color=colors['bolus'][:-2] + str(min((hypos_bolus_avg_after[i]) / 5, 1)) + ')',
                                                                id={'type': 'insights_hypos_card_bolus_avg_after', 'index': i}
                                                            ),
                                                        ],
                                                    ),
                                                ],
                                                width=4
                                            )
                                        ],
                                        style={'padding': '0%', 'margin': '0%'}
                                    ),
                                ],
                                style=hypos_styles[i],
                                id={'type': 'insights_hypos_pattern_card', 'index': i}
                            )
                            for i in range(0, num_insight_patterns)
                        ], width=9),

                    ################################################################################
                    # HYPO INSIGHTS: FILTER SIDEBAR
                    ################################################################################
                    dbc.Col(
                        html.Div(
                            [
                                ################################################################################
                                # HYPO INSIGHTS: SIDEBAR TITLE
                                ################################################################################
                                html.Div('FILTER', style={'font-family': font, 'font-size': 'small', 'padding': '0% 0% 0% 0%', 'font-weight': 'bold'}),

                                ################################################################################
                                # HYPO INSIGHTS: PATTERN GRAPHS OVERLAYED
                                ################################################################################
                                dbc.Row(
                                    dcc.Graph(
                                        figure=hypos_graph_all_curves,
                                        id='insights_hypos_graph_all_curves',
                                        config={
                                            'displayModeBar': False
                                        }
                                    ),
                                ),
                                html.Div(style={'padding': '0% 0 10%'}),

                                ################################################################################
                                # HYPO INSIGHTS: FILTER CHECKLIST TIME OF DAY
                                ################################################################################
                                html.Div(
                                    [
                                        dbc.Label("Time of day", style={'font-weight': 'bold'}),
                                        dbc.Checklist(
                                            options=[
                                                {"label": "Morning (6:00 - 11:00)", "value": 1},
                                                {"label": "Noon (11:00 - 16:00)", "value": 2},
                                                {"label": "Evening (16:00 - 24:00)", "value": 3},
                                                {"label": "Night (0:00 - 6:00)", "value": 4},
                                            ],
                                            value=[1, 2, 3, 4],
                                            input_checked_style={
                                                "backgroundColor": "#6c6c6c",
                                                "borderColor": "#6c6c6c",
                                            },
                                            id="insights_hypos_checklist_time_of_day",
                                        ),
                                    ]
                                ),
                                ################################################################################
                                # HYPO INSIGHTS: APPLY BUTTON
                                ################################################################################
                                dbc.Button("Apply",
                                           id='insights_hypos_filter_apply_btn',
                                           color="secondary",
                                           className="me-1",
                                           n_clicks=0,
                                           disabled=True,
                                           style={'margin': '10% 0'}
                                           ),
                            ],
                            style=SIDEBAR_STYLE,
                        ), width=3)
                ]
            )
            , label="Hypos", labelClassName='text-dark', activeTabClassName="fw-bold"
        ),
    ]
)
