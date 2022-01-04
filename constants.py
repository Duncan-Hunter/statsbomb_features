from feature_engineering import *


AGGREGATE_FUNCTIONS = {
    "normalised_location_count" : normalised_location_count,
    "normalised_count" : normalised_count,
    "count" : count_events,
    "shot_aerial_won_ratio" : shot_aerial_won_ratio,
    "shot_first_time_ratio" : shot_first_time_ratio,
    "direction_forward_ratio" : direction_forward_ratio,
    "direction_sideways_ratio" : direction_sideways_ratio,
    "direction_backward_ratio" : direction_backward_ratio,
    "distance_short_ratio" : distance_short_ratio,
    "distance_medium_ratio" : distance_medium_ratio,
    "distance_long_ratio" : distance_long_ratio,
    "success_rate" : success_rate,
    "high_pass_ratio" : high_pass_ratio,
    "cross_ratio" : cross_ratio
}


FEATURE_FUNCTIONS = {
    "direction_forward_ratio" : {
        "func" : pass_is_forward,
        "func_col" : "pass_angle",
        "out_col_name" : "pass_is_forwards",
        "null_val" : 0,
    },
    "direction_sideways_ratio" : {
        "func" : pass_is_sideways,
        "func_col" : "pass_angle",
        "out_col_name" : "pass_is_sideways",
        "null_val" : 0,
    },
    "direction_backward_ratio" : {
        "func" : pass_is_backward,
        "func_col" : "pass_angle",
        "out_col_name" : "pass_is_backwards",
        "null_val" : 0,
    },
    "distance_short_ratio" : {
        "func" : pass_is_short,
        "func_col" : "pass_length",
        "out_col_name" : "pass_is_short",
        "null_val" : 0,
    },
    "distance_medium_ratio" : {
        "func" : pass_is_medium,
        "func_col" : "pass_length",
        "out_col_name" : "pass_is_medium",
        "null_val" : 0,
    },
    "distance_long_ratio" : {
        "func" : pass_is_long,
        "func_col" : "pass_length",
        "out_col_name" : "pass_is_long",
        "null_val" : 0,
    },
}