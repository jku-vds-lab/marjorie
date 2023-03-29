

colors = dict(
    bg_very_low='rgb(251,90,82)',
    bg_low='rgb(255,140,126)',
    bg_target='rgb(120,211,168)',
    bg_high='rgb(188,155,233)',
    bg_very_high='rgb(139,99,213)',

    bolus='rgba(90,209,245, 1)',
    basal='rgba(0, 159, 219, 1)',
    carbs='rgba(255,213,114, 1)',
    background='rgba(248,249,250,1)',
    highlight='rgba(0, 0, 0, 0.2)',
    selected='rgba(230, 234, 238, 1)'
)

colors_agp = {
    'in_range_median': 'rgba(81,199,143, 1)',
    'in_range_75th': 'rgba(120,211,168, 0.5)',
    'in_range_90th': 'rgba(120,211,168, 0.2)',
    'above_range_75th': 'rgba(139,99,213, 0.4)',
    'above_range_90th': 'rgba(139,99,213, 0.2)',
    'under_range_75th': 'rgba(251,90,82, 0.5)',
    'under_range_90th': 'rgba(251,90,82, 0.2)',
}

colors_agp_bolus = {
    'in_range_median': 'rgba(90,209,245, 1)',
    'in_range_75th': 'rgba(90,209,245, 0.5)',
    'in_range_90th': 'rgba(90,209,245, 0.2)',
    'above_range_75th': 'rgba(90,209,245, 0.4)',
    'above_range_90th': 'rgba(90,209,245, 0.2)',
    'under_range_75th': 'rgba(90,209,245, 0.5)',
    'under_range_90th': 'rgba(90,209,245, 0.2)',
}

alpha = 0.5
colors_patterns = [
    'rgba(114, 158, 206, {})'.format(alpha),
    'rgba(255, 158, 74, {})'.format(alpha),
    'rgba(103, 191, 92, {})'.format(alpha),
    'rgba(237, 102, 93, {})'.format(alpha),
    'rgba(173, 139, 201, {})'.format(alpha),

    'rgba(114, 158, 206, {})'.format(alpha),
    'rgba(255, 158, 74, {})'.format(alpha),
    'rgba(103, 191, 92, {})'.format(alpha),
    'rgba(237, 102, 93, {})'.format(alpha),
    'rgba(173, 139, 201, {})'.format(alpha),

    'rgba(114, 158, 206, {})'.format(alpha),
    'rgba(255, 158, 74, {})'.format(alpha),
    'rgba(103, 191, 92, {})'.format(alpha),
    'rgba(237, 102, 93, {})'.format(alpha),
    'rgba(173, 139, 201, {})'.format(alpha),

    'rgba(114, 158, 206, {})'.format(alpha),
    'rgba(255, 158, 74, {})'.format(alpha),
    'rgba(103, 191, 92, {})'.format(alpha),
    'rgba(237, 102, 93, {})'.format(alpha),
    'rgba(173, 139, 201, {})'.format(alpha),

    'rgba(114, 158, 206, {})'.format(alpha),
    'rgba(255, 158, 74, {})'.format(alpha),
    'rgba(103, 191, 92, {})'.format(alpha),
    'rgba(237, 102, 93, {})'.format(alpha),
    'rgba(173, 139, 201, {})'.format(alpha),

    'rgba(114, 158, 206, {})'.format(alpha),
    'rgba(255, 158, 74, {})'.format(alpha),
    'rgba(103, 191, 92, {})'.format(alpha),
    'rgba(237, 102, 93, {})'.format(alpha),
    'rgba(173, 139, 201, {})'.format(alpha),

    'rgba(114, 158, 206, {})'.format(alpha),
    'rgba(255, 158, 74, {})'.format(alpha),
    'rgba(103, 191, 92, {})'.format(alpha),
    'rgba(237, 102, 93, {})'.format(alpha),
    'rgba(173, 139, 201, {})'.format(alpha),

    'rgba(114, 158, 206, {})'.format(alpha),
    'rgba(255, 158, 74, {})'.format(alpha),
    'rgba(103, 191, 92, {})'.format(alpha),
    'rgba(237, 102, 93, {})'.format(alpha),
    'rgba(173, 139, 201, {})'.format(alpha),

    'rgba(114, 158, 206, {})'.format(alpha),
    'rgba(255, 158, 74, {})'.format(alpha),
    'rgba(103, 191, 92, {})'.format(alpha),
    'rgba(237, 102, 93, {})'.format(alpha),
    'rgba(173, 139, 201, {})'.format(alpha),

    'rgba(114, 158, 206, {})'.format(alpha),
    'rgba(255, 158, 74, {})'.format(alpha),
    'rgba(103, 191, 92, {})'.format(alpha),
    'rgba(237, 102, 93, {})'.format(alpha),
    'rgba(173, 139, 201, {})'.format(alpha),

    'rgba(114, 158, 206, {})'.format(alpha),
    'rgba(255, 158, 74, {})'.format(alpha),
    'rgba(103, 191, 92, {})'.format(alpha),
    'rgba(237, 102, 93, {})'.format(alpha),
    'rgba(173, 139, 201, {})'.format(alpha),

    'rgba(114, 158, 206, {})'.format(alpha),
    'rgba(255, 158, 74, {})'.format(alpha),
    'rgba(103, 191, 92, {})'.format(alpha),
    'rgba(237, 102, 93, {})'.format(alpha),
    'rgba(173, 139, 201, {})'.format(alpha),
    'rgba(114, 158, 206, {})'.format(alpha),
    'rgba(255, 158, 74, {})'.format(alpha),
    'rgba(103, 191, 92, {})'.format(alpha),
    'rgba(237, 102, 93, {})'.format(alpha),
    'rgba(173, 139, 201, {})'.format(alpha),

    'rgba(114, 158, 206, {})'.format(alpha),
    'rgba(255, 158, 74, {})'.format(alpha),
    'rgba(103, 191, 92, {})'.format(alpha),
    'rgba(237, 102, 93, {})'.format(alpha),
    'rgba(173, 139, 201, {})'.format(alpha),
]

alpha = 0.2
colors_pattern_curves = [
    'rgba(114, 158, 206, {})'.format(alpha),
    'rgba(255, 158, 74, {})'.format(alpha),
    'rgba(103, 191, 92, {})'.format(alpha),
    'rgba(237, 102, 93, {})'.format(alpha),
    'rgba(173, 139, 201, {})'.format(alpha),

    'rgba(114, 158, 206, {})'.format(alpha),
    'rgba(255, 158, 74, {})'.format(alpha),
    'rgba(103, 191, 92, {})'.format(alpha),
    'rgba(237, 102, 93, {})'.format(alpha),
    'rgba(173, 139, 201, {})'.format(alpha),

    'rgba(114, 158, 206, {})'.format(alpha),
    'rgba(255, 158, 74, {})'.format(alpha),
    'rgba(103, 191, 92, {})'.format(alpha),
    'rgba(237, 102, 93, {})'.format(alpha),
    'rgba(173, 139, 201, {})'.format(alpha),

    'rgba(114, 158, 206, {})'.format(alpha),
    'rgba(255, 158, 74, {})'.format(alpha),
    'rgba(103, 191, 92, {})'.format(alpha),
    'rgba(237, 102, 93, {})'.format(alpha),
    'rgba(173, 139, 201, {})'.format(alpha),

    'rgba(114, 158, 206, {})'.format(alpha),
    'rgba(255, 158, 74, {})'.format(alpha),
    'rgba(103, 191, 92, {})'.format(alpha),
    'rgba(237, 102, 93, {})'.format(alpha),
    'rgba(173, 139, 201, {})'.format(alpha),

    'rgba(114, 158, 206, {})'.format(alpha),
    'rgba(255, 158, 74, {})'.format(alpha),
    'rgba(103, 191, 92, {})'.format(alpha),
    'rgba(237, 102, 93, {})'.format(alpha),
    'rgba(173, 139, 201, {})'.format(alpha),

    'rgba(114, 158, 206, {})'.format(alpha),
    'rgba(255, 158, 74, {})'.format(alpha),
    'rgba(103, 191, 92, {})'.format(alpha),
    'rgba(237, 102, 93, {})'.format(alpha),
    'rgba(173, 139, 201, {})'.format(alpha),

    'rgba(114, 158, 206, {})'.format(alpha),
    'rgba(255, 158, 74, {})'.format(alpha),
    'rgba(103, 191, 92, {})'.format(alpha),
    'rgba(237, 102, 93, {})'.format(alpha),
    'rgba(173, 139, 201, {})'.format(alpha),

    'rgba(114, 158, 206, {})'.format(alpha),
    'rgba(255, 158, 74, {})'.format(alpha),
    'rgba(103, 191, 92, {})'.format(alpha),
    'rgba(237, 102, 93, {})'.format(alpha),
    'rgba(173, 139, 201, {})'.format(alpha),

    'rgba(114, 158, 206, {})'.format(alpha),
    'rgba(255, 158, 74, {})'.format(alpha),
    'rgba(103, 191, 92, {})'.format(alpha),
    'rgba(237, 102, 93, {})'.format(alpha),
    'rgba(173, 139, 201, {})'.format(alpha),

    'rgba(114, 158, 206, {})'.format(alpha),
    'rgba(255, 158, 74, {})'.format(alpha),
    'rgba(103, 191, 92, {})'.format(alpha),
    'rgba(237, 102, 93, {})'.format(alpha),
    'rgba(173, 139, 201, {})'.format(alpha),

    'rgba(114, 158, 206, {})'.format(alpha),
    'rgba(255, 158, 74, {})'.format(alpha),
    'rgba(103, 191, 92, {})'.format(alpha),
    'rgba(237, 102, 93, {})'.format(alpha),
    'rgba(173, 139, 201, {})'.format(alpha),
    'rgba(114, 158, 206, {})'.format(alpha),
    'rgba(255, 158, 74, {})'.format(alpha),
    'rgba(103, 191, 92, {})'.format(alpha),
    'rgba(237, 102, 93, {})'.format(alpha),
    'rgba(173, 139, 201, {})'.format(alpha),

    'rgba(114, 158, 206, {})'.format(alpha),
    'rgba(255, 158, 74, {})'.format(alpha),
    'rgba(103, 191, 92, {})'.format(alpha),
    'rgba(237, 102, 93, {})'.format(alpha),
    'rgba(173, 139, 201, {})'.format(alpha),
]

color_secondary = '#5c636a'

colors_heatmap = [
    'rgb(221, 79, 55)',
    'rgb(227, 110, 91)',
    'rgb(218, 181, 166)',
    'rgb(176, 213, 188)',
    'rgb(179, 196, 205)',
    'rgb(138, 132, 200)',
    'rgb(104, 96, 184)'
]

targets_heatmap = [1, 54, 70, 100, 150, 180, 250, 350]
domain_heatmap = [target / targets_heatmap[-1] for target in targets_heatmap]


def get_prebolus_button_color(value):
    if value == 0:
        return 'rgba(0,0,0,0)'
    elif value < 0:
        return 'rgba(173, 63, 95,' + str(min(-value/30, 1)) + ')'
    else:
        return 'rgba(16, 125, 121,' + str(min(value/30, 1)) + ')'


def get_button_text_color(value):
    if value <= 0.5:
        return 'black'
    else:
        return 'white'
