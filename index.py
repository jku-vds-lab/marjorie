from datetime import timedelta, datetime, date
import dash
import numpy as np
from dash import dcc, html, ctx, MATCH

from aggregations import draw_seasonal_graph_day, draw_full_agp
from assets.styles import buttons_style_agp_date
from colors import colors_heatmap, colors, targets_heatmap, get_prebolus_button_color
from daily import draw_daily_plot
from insights import get_time_of_day_from_number, get_logs_meals, get_dataset, filter_function_time_of_day, filter_function_meal_size, get_curve_overview_plot, get_insight_data_meals, get_insight_data_hypos, \
    get_logs_hypos
from layout import app, layout_daily, layout_agp, layout_overview, layout_insights, create_horizon_graph
from dash_extensions.enrich import Output, DashProxy, Input, MultiplexerTransform, State, callback_context, ALL

from overview import draw_horizon_graph, get_daily_data, get_x_range_for_day, draw_overview_daily_curve_detailed
from statistics import get_tir_plot, get_statistics_day, get_statistics_days
from helpers import convert_datestring, get_df_between_dates, get_tir, get_statistics, check_timebox, get_log_indices, calculate_tir_time, get_df_of_date, get_mean_per_day
from preprocessing import dates, logs_sgv, date_max, date_min, start_date, end_date, sgv_array_for_agp, date_dict, logs_carbs, logs_insulin, logs_br_default, start_date_insights
from variables import num_horizon_graphs, num_insight_patterns, time_before_meal, time_after_meal, time_after_hypo, time_before_hypo
import re
import dash_bootstrap_components as dbc


@app.callback(Output("page-content", "children"), [Input("url", "pathname")], prevent_initial_call=True)
def render_page_content(pathname):
    if pathname == "/agp":
        return layout_agp
    if pathname == "/daily":
        return layout_daily
    elif pathname == "/insights":
        return layout_insights
    return layout_agp


################################################################################
# CALLBACKS DAILY
################################################################################

@app.callback(
    Output('stats_daily_sgv_ea1c', 'children'),
    Output('stats_daily_sgv_mean', 'children'),
    Output('stats_daily_sgv_std', 'children'),
    Output('stats_daily_very_low', 'children'),
    Output('stats_daily_low', 'children'),
    Output('stats_daily_target', 'children'),
    Output('stats_daily_high', 'children'),
    Output('stats_daily_very_high', 'children'),
    Output('stats_daily_very_low_time', 'children'),
    Output('stats_daily_low_time', 'children'),
    Output('stats_daily_target_time', 'children'),
    Output('stats_daily_high_time', 'children'),
    Output('stats_daily_very_high_time', 'children'),
    Output('stats_daily_carbs', 'children'),
    Output('stats_daily_bolus', 'children'),
    Output('stats_daily_basal', 'children'),
    Output('stats_daily_tir_graph', 'figure'),
    Output('daily_graph', 'figure'),
    Input('date_picker_daily', 'date')
)
def daily_date_picker_update(day):
    day = convert_datestring(day).date()
    stats, tir, carbs_sum, bolus_sum, basal_sum = get_statistics_day(day)

    return str(round(stats['ea1c'], 1)), \
           str(int(stats['mean'])), \
           str(int(stats['std'] / stats['mean'] * 100)), \
           str(tir[0]), \
           str(tir[1]), \
           str(tir[2]), \
           str(tir[3]), \
           str(tir[4]), \
           calculate_tir_time(tir[0]), \
           calculate_tir_time(tir[1]), \
           calculate_tir_time(tir[2]), \
           calculate_tir_time(tir[3]), \
           calculate_tir_time(tir[4]), \
           carbs_sum, \
           bolus_sum, \
           basal_sum, \
           get_tir_plot(tir), \
           draw_daily_plot(day)


@app.callback(
    Output('date_picker_daily', 'date'),
    Input('date_daily_back', 'n_clicks_timestamp'),
    State('date_picker_daily', 'date'),
)
def daily_date_button_backward(n_clicks, current_date):
    current_date = convert_datestring(current_date).date()
    return current_date - timedelta(days=1)


@app.callback(
    Output('date_picker_daily', 'date'),
    Input('date_daily_forward', 'n_clicks_timestamp'),
    State('date_picker_daily', 'date'),
)
def daily_date_button_forward(n_clicks, current_date):
    current_date = convert_datestring(current_date).date()
    return current_date + timedelta(days=1)


@app.callback(
    Output('daily_graph', 'figure'),
    Input('daily_graph', 'relayoutData'),
    State('date_picker_daily', 'date'),
)
def daily_semantic_zoom(zoom_data, default_date):
    if zoom_data:
        start = convert_datestring(zoom_data['xaxis.range[0]'])
        end = convert_datestring(zoom_data['xaxis.range[1]'])
        window_width = (end - start) / timedelta(hours=1)
    else:
        start = datetime.combine(datetime.strptime(default_date, '%Y-%m-%d').date(), datetime.min.time())
        end = start + timedelta(hours=24)
        window_width = 24

    return draw_daily_plot(start.date(), cutoff_value=window_width / 24 * 1.5, zoom_data=[start, end], bar_width_factor=window_width / 24 * 15 + 2)


################################################################################
# CALLBACKS AGP
################################################################################

@app.callback(
    Input('agp_date-picker-range', 'start_date'),
    Input('agp_date-picker-range', 'end_date'),
    Output('agp_stats_sgv_ea1c', 'children'),
    Output('agp_stats_sgv_mean', 'children'),
    Output('agp_stats_sgv_std', 'children'),
    Output('agp_stats_very_low', 'children'),
    Output('agp_stats_low', 'children'),
    Output('agp_stats_target', 'children'),
    Output('agp_stats_high', 'children'),
    Output('agp_stats_very_high', 'children'),
    Output('agp_stats_very_low_time', 'children'),
    Output('agp_stats_low_time', 'children'),
    Output('agp_stats_target_time', 'children'),
    Output('agp_stats_high_time', 'children'),
    Output('agp_stats_very_high_time', 'children'),
    Output('agp_stats_carbs', 'children'),
    Output('agp_stats_bolus', 'children'),
    Output('agp_stats_basal', 'children'),
    Output('agp_stats_tir_graph', 'figure'),
    Output('agp_graph', 'figure'),
    prevent_initial_call=True
)
def agp_update_dates(date_start, date_end):
    stats, tir, carbs_sum, bolus_sum, basal_sum = get_statistics_days(date_start, date_end)
    return str(round(stats['ea1c'], 1)), \
           str(int(stats['mean'])), \
           str(int(stats['std'] / stats['mean'] * 100)), \
           str(tir[0]), \
           str(tir[1]), \
           str(tir[2]), \
           str(tir[3]), \
           str(tir[4]), \
           calculate_tir_time(tir[0]), \
           calculate_tir_time(tir[1]), \
           calculate_tir_time(tir[2]), \
           calculate_tir_time(tir[3]), \
           calculate_tir_time(tir[4]), \
           carbs_sum, \
           bolus_sum, \
           basal_sum, \
           get_tir_plot(tir), \
           draw_full_agp(date_start, date_end), \
        # draw_seasonal_graph_day(date_start, date_end)


@app.callback(
    Input({'type': 'agp_quick_date_button', 'index': ALL}, 'n_clicks'),
    Output('agp_date-picker-range', 'start_date'),
    Output('agp_date-picker-range', 'end_date'),
    prevent_initial_call=True
)
def agp_quick_date_buttons(n_clicks):
    triggered_id = ctx.triggered_id
    return date_max.date() - timedelta(days=triggered_id['index'] * 7), date_max.date()


@app.callback(
    Input('agp_explore_button', 'n_clicks'),
    Output("page-content", "children"),
    prevent_initial_call=True
)
def explore_days_in_detail_button(_):
    return layout_overview


@app.callback(
    Input('agp_open_weekly_stats', 'n_clicks'),
    Output('agp_modal', 'is_open')
)
def agp_open_weekly_stats(_):
    return True


@app.callback(
    [Input('agp_weekday_button-{}-'.format(i), 'n_clicks') for i in range(7)],
    [State('agp_weekday_button-{}-'.format(i), 'active') for i in range(7)],
    State('agp_date-picker-range', 'start_date'),
    State('agp_date-picker-range', 'end_date'),
    Output('agp_stats_sgv_ea1c', 'children'),
    Output('agp_stats_sgv_mean', 'children'),
    Output('agp_stats_sgv_std', 'children'),
    Output('agp_stats_very_low', 'children'),
    Output('agp_stats_low', 'children'),
    Output('agp_stats_target', 'children'),
    Output('agp_stats_high', 'children'),
    Output('agp_stats_very_high', 'children'),
    Output('agp_stats_very_low_time', 'children'),
    Output('agp_stats_low_time', 'children'),
    Output('agp_stats_target_time', 'children'),
    Output('agp_stats_high_time', 'children'),
    Output('agp_stats_very_high_time', 'children'),
    Output('agp_stats_carbs', 'children'),
    Output('agp_stats_bolus', 'children'),
    Output('agp_stats_basal', 'children'),
    Output('agp_stats_tir_graph', 'figure'),
    Output('agp_graph', 'figure'),
    [Output('agp_weekday_button-{}-'.format(i), 'active') for i in range(7)],
    prevent_initial_call=True

)
def agp_weekday_buttons(*args):
    state = args[7:14]
    date_start = args[-2]
    date_end = args[-1]
    context = callback_context.triggered
    if context:
        clicked = int(re.search('-(.*)-', context[0]['prop_id']).group(1))
        state = [dash.no_update] * clicked + [not state[clicked]] + [dash.no_update] * (7 - clicked - 1)
        weekday_filter = np.where(np.array(state))[0]
        agp_figure = draw_full_agp(date_start, date_end, weekday_filter)
        stats, tir, carbs_sum, bolus_sum, basal_sum = get_statistics_days(date_start, date_end, weekday_filter)
        stats_ea1c = str(round(stats['ea1c'], 1))
        stats_mean = str(int(stats['mean']))
        stats_std = str(int(stats['std'] / stats['mean'] * 100))
        stats_tir_0 = str(tir[0])
        stats_tir_1 = str(tir[1])
        stats_tir_2 = str(tir[2])
        stats_tir_3 = str(tir[3])
        stats_tir_4 = str(tir[4])
        stats_time_0 = calculate_tir_time(tir[0])
        stats_time_1 = calculate_tir_time(tir[1])
        stats_time_2 = calculate_tir_time(tir[2])
        stats_time_3 = calculate_tir_time(tir[3])
        stats_time_4 = calculate_tir_time(tir[4])
        stats_carbs = carbs_sum
        stats_bolus = bolus_sum
        stats_basal = basal_sum
        stats_plot = get_tir_plot(tir)

    else:
        state = [dash.no_update] * 7
        agp_figure = dash.no_update
        stats_ea1c = dash.no_update
        stats_mean = dash.no_update
        stats_std = dash.no_update
        stats_tir_0 = dash.no_update
        stats_tir_1 = dash.no_update
        stats_tir_2 = dash.no_update
        stats_tir_3 = dash.no_update
        stats_tir_4 = dash.no_update
        stats_time_0 = dash.no_update
        stats_time_1 = dash.no_update
        stats_time_2 = dash.no_update
        stats_time_3 = dash.no_update
        stats_time_4 = dash.no_update
        stats_carbs = dash.no_update
        stats_bolus = dash.no_update
        stats_basal = dash.no_update
        stats_plot = dash.no_update

    return stats_ea1c, \
           stats_mean, \
           stats_std, \
           stats_tir_0, \
           stats_tir_1, \
           stats_tir_2, \
           stats_tir_3, \
           stats_tir_4, \
           stats_time_0, \
           stats_time_1, \
           stats_time_2, \
           stats_time_3, \
           stats_time_4, \
           stats_carbs, \
           stats_bolus, \
           stats_basal, \
           stats_plot, \
           agp_figure, \
           *state


################################################################################
# CALLBACKS OVERVIEW
################################################################################

@app.callback(
    Output({'type': 'horizon_graph', 'index': MATCH}, 'figure'),
    Output({"type": "btn_horizon_graph_expand", "index": MATCH}, 'children'),
    Input({"type": "btn_horizon_graph_expand", "index": MATCH}, 'n_clicks'),
    State({'type': 'horizon_date_info', 'index': MATCH}, 'children'),
    State({'type': 'horizon_graph', 'index': MATCH}, 'figure'),
    prevent_initial_call=True
)
def overview_expand_button_click(_, day, figure):
    day = datetime.strptime(day, '%d/%m/%Y').date()
    if figure['layout']['height'] > 100:
        graph = draw_horizon_graph(*get_daily_data(day), x_range=get_x_range_for_day(day))
        button = html.Span([html.I(className="fas fa-caret-down fa-2x")])
    else:
        graph = draw_overview_daily_curve_detailed(*get_daily_data(day), x_range=get_x_range_for_day(day))
        button = html.Span([html.I(className="fas fa-caret-up fa-2x")])
    return graph, button


@app.callback(
    Input('overview_date-picker-range', 'start_date'),
    Input('overview_date-picker-range', 'end_date'),
    Output('overview_stats_sgv_ea1c', 'children'),
    Output('overview_stats_sgv_mean', 'children'),
    Output('overview_stats_sgv_std', 'children'),
    Output('overview_stats_very_low', 'children'),
    Output('overview_stats_low', 'children'),
    Output('overview_stats_target', 'children'),
    Output('overview_stats_high', 'children'),
    Output('overview_stats_very_high', 'children'),
    Output('overview_stats_very_low_time', 'children'),
    Output('overview_stats_low_time', 'children'),
    Output('overview_stats_target_time', 'children'),
    Output('overview_stats_high_time', 'children'),
    Output('overview_stats_very_high_time', 'children'),
    Output('overview_stats_carbs', 'children'),
    Output('overview_stats_bolus', 'children'),
    Output('overview_stats_basal', 'children'),
    Output('overview_stats_tir_graph', 'figure'),
    Output('overview_agp_graph', 'figure'),
    Output('overview_horizon_graphs', 'children')
)
def overview_update_dates(date_start, date_end):
    stats, tir, carbs_sum, bolus_sum, basal_sum = get_statistics_days(date_start, date_end)
    date_start_dt = convert_datestring(date_start).date()
    date_end_dt = convert_datestring(date_end).date()
    days_horizon_graphs = [date for date in dates if (date_start_dt <= date <= date_end_dt)]

    horizon_card_content = []
    for id, date in enumerate(days_horizon_graphs):
        horizon_card_content.append(create_horizon_graph(id, date))

    return str(round(stats['ea1c'], 1)), \
           str(int(stats['mean'])), \
           str(int(stats['std'] / stats['mean'] * 100)), \
           str(tir[0]), \
           str(tir[1]), \
           str(tir[2]), \
           str(tir[3]), \
           str(tir[4]), \
           calculate_tir_time(tir[0]), \
           calculate_tir_time(tir[1]), \
           calculate_tir_time(tir[2]), \
           calculate_tir_time(tir[3]), \
           calculate_tir_time(tir[4]), \
           carbs_sum, \
           bolus_sum, \
           basal_sum, \
           get_tir_plot(tir), \
           draw_seasonal_graph_day(date_start, date_end), \
           horizon_card_content


@app.callback(
    Input({'type': 'overview_quick_date_button', 'index': ALL}, 'n_clicks'),
    Output('overview_date-picker-range', 'start_date'),
    Output('overview_date-picker-range', 'end_date'),
    prevent_initial_call=True
)
def overview_quick_date_buttons(n_clicks):
    triggered_id = ctx.triggered_id
    return date_max.date() - timedelta(days=triggered_id['index'] * 7), date_max.date()


@app.callback(
    Input('overview_open_weekly_stats', 'n_clicks'),
    Output('overview_modal', 'is_open'),
    prevent_initial_call=True
)
def overview_open_weekly_stats(_):
    return True


# @app.callback(
#     [Input('overview_weekday_button-{}-'.format(i), 'n_clicks') for i in range(7)],
#     [State('overview_weekday_button-{}-'.format(i), 'active') for i in range(7)],
#     State('overview_date-picker-range', 'start_date'),
#     State('overview_date-picker-range', 'end_date'),
#     Output('overview_stats_sgv_ea1c', 'children'),
#     Output('overview_stats_sgv_mean', 'children'),
#     Output('overview_stats_sgv_std', 'children'),
#     Output('overview_stats_very_low', 'children'),
#     Output('overview_stats_low', 'children'),
#     Output('overview_stats_target', 'children'),
#     Output('overview_stats_high', 'children'),
#     Output('overview_stats_very_high', 'children'),
#     Output('overview_stats_very_low_time', 'children'),
#     Output('overview_stats_low_time', 'children'),
#     Output('overview_stats_target_time', 'children'),
#     Output('overview_stats_high_time', 'children'),
#     Output('overview_stats_very_high_time', 'children'),
#     Output('overview_stats_carbs', 'children'),
#     Output('overview_stats_bolus', 'children'),
#     Output('overview_stats_basal', 'children'),
#     Output('overview_stats_tir_graph', 'figure'),
#     Output('overview_agp_graph', 'figure'),
#     *[Output('pattern_detail_agp_div_{}'.format(i), 'style') for i in range(num_horizon_graphs)],
#     [Output('overview_weekday_button-{}-'.format(i), 'active') for i in range(7)],
#     prevent_initial_call=True
# )
# def overview_weekday_buttons(*args):
#     state = args[7:14]
#     date_start = args[-2]
#     date_end = args[-1]
#     context = callback_context.triggered
#     if context:
#         clicked = int(re.search('-(.*)-', context[0]['prop_id']).group(1))
#         state = [dash.no_update] * clicked + [not state[clicked]] + [dash.no_update] * (7 - clicked - 1)
#         weekday_filter = np.where(np.array(state))[0]
#         # agp_figure = draw_full_agp(date_start, date_end, weekday_filter)
#         stats, tir, carbs_sum, bolus_sum, basal_sum = get_statistics_days(date_start, date_end, weekday_filter)
#         stats_ea1c = str(round(stats['ea1c'], 1))
#         stats_mean = str(int(stats['mean']))
#         stats_std = str(int(stats['std'] / stats['mean'] * 100))
#         stats_tir_0 = str(tir[0])
#         stats_tir_1 = str(tir[1])
#         stats_tir_2 = str(tir[2])
#         stats_tir_3 = str(tir[3])
#         stats_tir_4 = str(tir[4])
#         stats_time_0 = calculate_tir_time(tir[0])
#         stats_time_1 = calculate_tir_time(tir[1])
#         stats_time_2 = calculate_tir_time(tir[2])
#         stats_time_3 = calculate_tir_time(tir[3])
#         stats_time_4 = calculate_tir_time(tir[4])
#         stats_carbs = carbs_sum
#         stats_bolus = bolus_sum
#         stats_basal = basal_sum
#         stats_plot = get_tir_plot(tir)
#
#         overview_agp_graph = draw_seasonal_graph_day(date_start, date_end, weekday_filter)
#
#         date_start_dt = convert_datestring(date_start).date()
#         date_end_dt = convert_datestring(date_end).date()
#         days_horizon_graphs = [date for date in dates if (date_start_dt <= date <= date_end_dt)]
#         styles = [{'display': 'inline'} if (date.weekday() in weekday_filter) else {'display': 'none'} for date in days_horizon_graphs] + [{'display': 'none'}] * (num_horizon_graphs - len(days_horizon_graphs))
#
#     else:
#         state = [dash.no_update] * 7
#         stats_ea1c = dash.no_update
#         stats_mean = dash.no_update
#         stats_std = dash.no_update
#         stats_tir_0 = dash.no_update
#         stats_tir_1 = dash.no_update
#         stats_tir_2 = dash.no_update
#         stats_tir_3 = dash.no_update
#         stats_tir_4 = dash.no_update
#         stats_time_0 = dash.no_update
#         stats_time_1 = dash.no_update
#         stats_time_2 = dash.no_update
#         stats_time_3 = dash.no_update
#         stats_time_4 = dash.no_update
#         stats_carbs = dash.no_update
#         stats_bolus = dash.no_update
#         stats_basal = dash.no_update
#         stats_plot = dash.no_update
#         overview_agp_graph = dash.no_update
#         styles = dash.no_update
#
#     return stats_ea1c, \
#            stats_mean,\
#            stats_std,\
#            stats_tir_0,\
#            stats_tir_1,\
#            stats_tir_2,\
#            stats_tir_3,\
#            stats_tir_4,\
#            stats_time_0,\
#            stats_time_1,\
#            stats_time_2,\
#            stats_time_3,\
#            stats_time_4,\
#            stats_carbs,\
#            stats_bolus,\
#            stats_basal,\
#            stats_plot,\
#            overview_agp_graph, \
#            *styles, \
#            *state


################################################################################
# CALLBACKS INSIGHTS
################################################################################

@app.callback(
    Input('insights_meals_filter_apply_btn', 'n_clicks'),
    Input("insights_meals_checklist_time_of_day", 'value'),
    Input('insights_meals_range_slider_meal_size', 'value'),
    Output('insights_meals_graph_all_curves', 'figure'),
    Output('insights_meals_filter_apply_btn', 'disabled'),
    *[Output('insights_meals_bar_graph_{}'.format(i), 'figure') for i in range(num_insight_patterns)],
    *[Output('insights_meals_overview_graph_{}'.format(i), 'figure') for i in range(num_insight_patterns)],
    Output({'type': 'insights_meals_sgv_before', 'index': ALL}, 'children'),
    Output({'type': 'insights_meals_sgv_after', 'index': ALL}, 'children'),
    Output({'type': 'insights_meals_interval', 'index': ALL}, 'children'),
    Output({'type': 'insights_meals_meal_size', 'index': ALL}, 'children'),
    Output({'type': 'insights_meals_bolus', 'index': ALL}, 'children'),
    Output({'type': 'insights_meals_factor', 'index': ALL}, 'children'),
    Output({'type': 'insights_meals_card_sgv_before', 'index': ALL}, 'color'),
    Output({'type': 'insights_meals_card_sgv_after', 'index': ALL}, 'color'),
    Output({'type': 'insights_meals_card_interval', 'index': ALL}, 'color'),
    Output({'type': 'insights_meals_card_meal_size', 'index': ALL}, 'color'),
    Output({'type': 'insights_meals_card_bolus', 'index': ALL}, 'color'),
    Output({'type': 'insights_meals_card_factor', 'index': ALL}, 'color'),
    Output({'type': 'insights_meals_pattern_card', 'index': ALL}, 'style'),
)
def update_insights_meals(_, time_of_day_filter, meal_size_filter):
    time_of_day_filter = get_time_of_day_from_number(time_of_day_filter)
    triggered_id = ctx.triggered_id

    if triggered_id == 'insights_meals_filter_apply_btn':
        print('##############################################')
        n_clusters_, graphs_meal_overview, graphs_all_curves, graphs_insights_meals, start_bgs, time_between, carbs_sums, end_bgs, bolus_sums = get_insight_data_meals(
            filter_time_of_day=time_of_day_filter,
            filter_meal_size=meal_size_filter)

        color_sgv_before = [colors_heatmap[list(np.array(targets_heatmap) > bg).index(True) - 1] for bg in start_bgs]
        color_sgv_after = [colors_heatmap[list(np.array(targets_heatmap) > bg).index(True) - 1] for bg in end_bgs]
        color_time_between = [get_prebolus_button_color(item) for item in time_between]
        color_meal_size = [colors['carbs'][:-2] + str(min((item - 20) / 90, 1)) + ')' for item in carbs_sums]
        color_bolus = [colors['bolus'][:-2] + str(min((item - 5) / 14, 1)) + ')' for item in bolus_sums]
        color_factor = ['rgba(157, 164, 169,' + str(min((carbs / bolus) / 10, 1)) + ')' for carbs, bolus in zip(carbs_sums, bolus_sums)]

        carbs_sum = [str(round(c)) for c in carbs_sums]
        bolus_sum = [str(round(b, 1)) for b in bolus_sums]
        factors = [str(round(c / b)) for c, b in zip(carbs_sums, bolus_sums)]

        styles = [{'display': 'inline'}] * n_clusters_ + [{'display': 'none'}] * (num_insight_patterns - n_clusters_)

        n_patterns_text = '{} patterns were found.'.format(n_clusters_)
        print(n_patterns_text)
        return dash.no_update, True, *graphs_meal_overview, *graphs_insights_meals, start_bgs, end_bgs, time_between, carbs_sum, bolus_sum, factors, color_sgv_before, color_sgv_after, color_time_between, \
               color_meal_size, color_bolus, color_factor, styles
    else:
        logs_meals = get_logs_meals(start_date_insights, end_date, time_before_meal, time_after_meal)
        dataset_unfiltered, _ = get_dataset(logs_meals)
        logs_meals = filter_function_time_of_day(logs_meals, time_of_day_filter)
        logs_meals = filter_function_meal_size(logs_meals, meal_size_filter)

        dataset_clusters, _ = get_dataset(logs_meals)
        figure = get_curve_overview_plot(dataset_clusters, dataset_unfiltered)

        no_update = [dash.no_update] * num_insight_patterns
        return figure, False, *no_update, *no_update, no_update, no_update, no_update, no_update, no_update, no_update, no_update, no_update, no_update, no_update, no_update, no_update, no_update


@app.callback(
    Input('insights_hypos_filter_apply_btn', 'n_clicks'),
    Input('insights_hypos_checklist_time_of_day', 'value'),
    Output('insights_hypos_graph_all_curves', 'figure'),
    Output('insights_hypos_filter_apply_btn', 'disabled'),
    *[Output('insights_hypos_bar_graph_{}'.format(i), 'figure') for i in range(num_insight_patterns)],
    *[Output('insights_hypos_overview_graph_{}'.format(i), 'figure') for i in range(num_insight_patterns)],
    Output({'type': 'insights_hypos_sgv_before', 'index': ALL}, 'children'),
    Output({'type': 'insights_hypos_sgv_after', 'index': ALL}, 'children'),
    Output({'type': 'insights_hypos_carb_avg_before', 'index': ALL}, 'children'),
    Output({'type': 'insights_hypos_carb_avg_after', 'index': ALL}, 'children'),
    Output({'type': 'insights_hypos_bolus_avg_before', 'index': ALL}, 'children'),
    Output({'type': 'insights_hypos_bolus_avg_after', 'index': ALL}, 'children'),
    Output({'type': 'insights_hypos_card_sgv_before', 'index': ALL}, 'color'),
    Output({'type': 'insights_hypos_card_sgv_after', 'index': ALL}, 'color'),
    Output({'type': 'insights_hypos_card_carb_avg_before', 'index': ALL}, 'color'),
    Output({'type': 'insights_hypos_card_carb_avg_after', 'index': ALL}, 'color'),
    Output({'type': 'insights_hypos_card_bolus_avg_before', 'index': ALL}, 'color'),
    Output({'type': 'insights_hypos_card_bolus_avg_after', 'index': ALL}, 'color'),
    Output({'type': 'insights_hypos_pattern_card', 'index': ALL}, 'style'),
)
def update_insights_hypos(_, time_of_day_filter):
    time_of_day_filter = get_time_of_day_from_number(time_of_day_filter)
    triggered_id = ctx.triggered_id

    if triggered_id == 'insights_hypos_filter_apply_btn':
        hypos_n_clusters_, hypos_bar_graphs, hypos_graph_all_curves, hypos_graphs_insights, hypos_start_bgs, hypos_end_bgs, hypos_carb_avg_before, hypos_carb_avg_after, hypos_bolus_avg_before, \
        hypos_bolus_avg_after = get_insight_data_hypos(filter_time_of_day=time_of_day_filter)

        color_sgv_before = [colors_heatmap[list(np.array(targets_heatmap) > bg).index(True) - 1] for bg in hypos_start_bgs]
        color_sgv_after = [colors_heatmap[list(np.array(targets_heatmap) > bg).index(True) - 1] for bg in hypos_end_bgs]
        color_carb_avg_before = [colors['carbs'][:-2] + str(min((item - 20) / 60, 1)) + ')' for item in hypos_carb_avg_before]
        color_carb_avg_after = [colors['carbs'][:-2] + str(min((item - 20) / 60, 1)) + ')' for item in hypos_carb_avg_after]
        color_bolus_avg_before = [colors['bolus'][:-2] + str(min((item - 5) / 5, 1)) + ')' for item in hypos_bolus_avg_before]
        color_bolus_avg_after = [colors['bolus'][:-2] + str(min((item - 5) / 5, 1)) + ')' for item in hypos_bolus_avg_after]

        hypos_carb_avg_before = [str(round(c)) for c in hypos_carb_avg_before]
        hypos_carb_avg_after = [str(round(c)) for c in hypos_carb_avg_after]
        hypos_bolus_avg_before = [str(round(b, 1)) for b in hypos_bolus_avg_before]
        hypos_bolus_avg_after = [str(round(b, 1)) for b in hypos_bolus_avg_after]

        styles = [{'display': 'inline'}] * hypos_n_clusters_ + [{'display': 'none'}] * (num_insight_patterns - hypos_n_clusters_)

        n_patterns_text = '{} patterns were found.'.format(hypos_n_clusters_)
        print(n_patterns_text)
        return dash.no_update, True, *hypos_bar_graphs, *hypos_graphs_insights, hypos_start_bgs, hypos_end_bgs, hypos_carb_avg_before, hypos_carb_avg_after, hypos_bolus_avg_before, hypos_bolus_avg_after, color_sgv_before, color_sgv_after, color_carb_avg_before, \
               color_carb_avg_after, color_bolus_avg_before, color_bolus_avg_after, styles
    else:
        logs_hypos, hypo_starts = get_logs_hypos(start_date_insights, end_date, time_before_hypo, time_after_hypo)
        dataset_unfiltered, _ = get_dataset(logs_hypos)
        print(time_of_day_filter)
        logs_hypos = filter_function_time_of_day(logs_hypos, time_of_day_filter)

        dataset_clusters, _ = get_dataset(logs_hypos)
        figure = get_curve_overview_plot(dataset_clusters, dataset_unfiltered, insights_type='hypos')

        no_update = [dash.no_update] * num_insight_patterns
        return figure, False, *no_update, *no_update, no_update, no_update, no_update, no_update, no_update, no_update, no_update, no_update, no_update, no_update, no_update, no_update, no_update


if __name__ == '__main__':
    app.run_server(host='0.0.0.0', port=8050, debug=False)
