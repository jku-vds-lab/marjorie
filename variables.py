target_range = [70, 180]
target_range_extended = [54, 70, 180, 250]
target_range_dict = {
    'very low': 54,
    'low': 70,
    'high': 180,
    'very high': 250
}
num_motifs = 20
num_segments = 12
window_size = 2
n_clusters = 4
num_horizon_graphs = 30
num_insight_patterns = 15
num_insight_details = 50
time_before_meal = 1
time_after_meal = 3
time_before_hypo = 2
time_after_hypo = 2
initial_number_of_days = 14
n_filters = 10
font = 'Verdana, sans-serif'


morning = {n: 'morning' for n in range(6, 11)}
noon = {n: 'noon' for n in range(11, 16)}
evening = {n: 'evening' for n in range(16, 24)}
night = {n: 'night' for n in range(0, 6)}
times_of_day = {**morning, **noon, **evening, **night}