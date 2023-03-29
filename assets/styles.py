font = 'Verdana, sans-serif'
from colors import colors, color_secondary

SIDEBAR_STYLE = {
    "position": "fixed",
    "top": 0,
    "right": 0,
    # "bottom": 0,
    "float": "right",
    # "width": "29rem",
    # "padding": "2rem 1rem",
    "background-color": "#f8f9fa",
    "height": "100%",
    # "overflow": "scroll",
    'overflow-x': 'hidden',
    'overflow-y': 'hidden',
    "padding": "2rem 2rem",
    # 'z-index': '12000'
}

SIDEBAR_STYLE_LEFT = {
    "position": "fixed",
    "top": 0,
    "left": 0,
    "bottom": 0,
    "width": "14rem",
    "padding": "2rem 1rem",
    "background-color": "#f8f9fa",
    'z-index': '12000'
}

CONTENT_STYLE = {
    "margin-left": "0rem",
    "margin-right": "2rem",
    "padding": "2rem 0rem 0rem 50px",  # ice: 5rem
}

# SIDEBAR_LEFT_STYLE = {
#     "position": "absolute",
#     "top": 0,
#     "left": 0,
#     # "bottom": 0,
#     "float": "right",
#     "width": "4rem",
#     # "padding": "2rem 1rem",
#     "background-color": "#f8f9fa",
#     "height": "100%",
#     # "overflow": "scroll",
# }

pattern_details_style = {
    # "overflow": "scroll",
    "maxHeight": "20rem",
    'position': 'relative',
    'top': '6rem',
    'left': '0em'
}

# the styles for the main content position it to the right of the sidebar and
# add some padding.
timeline_div_style = {
    "position": "absolute",
    "top": 80,
    "left": '6rem',
    "float": "left",
    'overflow': 'hidden'
    # "width": "100vh",
    # "overflow": "scroll",
}

overview_div_style = {
    "position": "absolute",
    "top": '0',
    "left": '8rem',
    "float": "left",
    'overflow-y': 'hidden',
    'overflow-x': 'hidden',
    'padding': '0rem 0rem 0rem 0rem'
    # "width": "100vh",
    # "overflow": "scroll",
}

daily_div_style = {
    "position": "absolute",
    "top": '5rem',
    "left": '6rem',
    "float": "left",
    'overflow-y': 'hidden',
    'overflow-x': 'hidden',
    'padding': '0rem 0rem 0rem 0rem'
    # "width": "100vh",
    # "overflow": "scroll",
}

seasonal_div_style = {
    "position": "absolute",
    "top": 0,
    "left": '6rem',
    "float": "left",
    'overflow-y': 'hidden'
    # "width": "100vh",
    # "overflow": "scroll",
}

tabs_styles = {
    "position": "absolute",
    'height': '10rem',
    # "top": 80,
    "left": '10rem',
    # 'height': '44px',
    "float": "left",
    # 'horizontalAlign': 'left'
}
tab_style = {
    'backgroundColor': colors['selected'],
    'fontColor': 'black',
    'font-size': 'small',
    'font-family': font,
    # 'font-weight': 'bold',
    'padding': '1rem 1rem',
    'margin': 'auto',
    'position': 'sticky', 'z-index': '1000', 'top': '0', 'height': '4rem',
    'border-style': 'solid',
    'border-color': 'rgba(0,0,0,0)',
    # 'align-items': 'center',
}

tab_selected_style = {
    'backgroundColor': colors['background'],
    'font-size': 'small',
    'fontColor': 'black',
    'font-weight': 'bold',
    'padding': '1rem 1rem',
    'font-family': font,
    'position': 'sticky', 'z-index': '1000', 'top': '0', 'height': '4rem',
    'border-style': 'solid',
    'border-color': 'rgba(0,0,0,0)',
    'align-items': 'center',
}

tab_style_main_view = {
    'backgroundColor': colors['selected'],
    'fontColor': 'black',
    'font-size': 'small',
    'font-family': font,
    'width': '9rem',
    # 'font-weight': 'bold',
    'padding': '1rem 0rem',
    'position': 'sticky', 'z-index': '1000', 'top': '0', 'height': '4rem',
    'border-style': 'solid',
    'border-color': 'rgba(0,0,0,0)',
    'align-items': 'center',
}

tab_selected_style_main_view = {
    'backgroundColor': 'white',
    'font-size': 'small',
    'fontColor': 'black',
    'font-weight': 'bold',
    'width': '9rem',
    'padding': '1rem 0rem',
    'font-family': font,
    'position': 'sticky', 'z-index': '1000', 'top': '0', 'height': '4rem',
    'border-style': 'solid',
    'border-color': 'rgba(0,0,0,0)',
    'align-items': 'center',
}

style_motif_summary = {
    # 'border-style': 'solid',
    # 'border-color': 'white',
    'padding': '0rem 0rem 0rem 0rem'
}

buttons_style_icon = {
    'height': '4rem',
    'width': '2rem',
    'text-align': 'center',
    'line-height': '50%',
    # 'float': 'left',
    # 'text-transform': 'none',
    # 'align-items': 'center',
    # 'display': 'flex',
    # 'justify-content': 'center',
    # 'position': 'relative',
    # 'left': '10%',
    # 'padding': '0rem 0rem',
    # 'line-height': '2rem',
    # 'line-height': '1rem',
    # 'font-size': '10px'
}


buttons_style_aggregation = {
    'height': '2rem',
    'width': '2rem',
    # 'float': 'left',
    'text-transform': 'none',
    'text-align': 'center',
    # 'position': 'relative',
    # 'left': '10%',
    'padding': '0rem 0rem',
    'line-height': '2rem',
    # 'line-height': '1rem',
    # 'font-size': '10px'
}

buttons_style_aggregation_clicked = {
    'height': '2rem',
    'width': '2rem',
    'text-transform': 'none',
    # 'float': 'left',
    'background-color': colors['selected'],
    'text-align': 'center',
    # 'position': 'relative',
    # 'left': '10%',
    'padding': '0rem 0rem',
    'line-height': '2rem',
    # 'line-height': '1rem',
    # 'font-size': '10px'
}

buttons_style_motif = {
    # 'word-break': 'keep-all',
    'white-space': 'normal',
    'border': 'none',
    'background': 'none',
    # "padding": "0rem 0rem 3rem 4rem",
    'text-transform': 'none',
    # 'height': '1em',
    'line-height': '1em',
    'font-family': font,
    'width': '12rem',
    # 'font-size': '1rem'
}

buttons_style_agp_date = {
    # 'word-break': 'keep-all',
    'white-space': 'normal',
    'border': 'none',
    'background': 'none',
    # "padding": "0rem 0rem 3rem 4rem",
    'text-transform': 'none',
    # 'height': '1em',
    'line-height': '1em',
    'font-family': font,
    'width': '12rem',
    'position': 'absolute', 'left': '-30px',  # ice: -2rem
    # 'font-size': '1rem'
}

buttons_style_agp_expand = {
    # 'word-break': 'keep-all',
    # 'white-space': 'normal',
    # 'border': 'none',
    # 'background': 'none',
    # "padding": "0rem 0rem 3rem 4rem",
    # 'text-transform': 'none',
    'height': '10px',
    'line-height': '10px',
    # 'font-family': font,
    'width': '10px',
    # 'position': 'absolute', 'left': '-30px',  # ice: -2rem
    # 'font-size': '1rem'
}

style_tir_target_text = {
    # 'font-family': font,
                         'font-weight': 'bold',
                         'font-size': 'medium',
                         'padding': '0% 15%',
                         'float': 'right'}

style_tir_text = {
    # 'font-family': font,
                  'font-size': 'small',
                  'padding': '0% 15%',
                  'float': 'right'}


style_link = {
    'height': '2rem',
    'text-transform': 'none',
    'text-decoration': 'none',
    'color': color_secondary,
    'font-family': font
}