from sklearn.cluster import KMeans
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple, List, Any, Callable
from argparse import ArgumentParser
from statsbombpy import sb
from yaml import load, SafeLoader, parse
from functools import reduce
import os
from time import time


def load_and_filter(competition_id: int,
                    season_id: int,
                    team_name: str = None) -> pd.DataFrame:
    """Loads and filters statsbomb matches."""
    matches = sb.matches(competition_id=competition_id,
                         season_id=season_id).sort_values(by="match_date")
    if team_name is not None:
        matches = matches[(matches.home_team == team_name) |
                          (matches.away_team == team_name)]
    if matches.empty:
        raise ValueError(
            f"""Matches is empty. Please check combination of {competition_id = }, {season_id = }, and {team_name = }"""
        )
    return matches


def filter_and_apply(events: pd.DataFrame,
                     _types: List[str],
                     func,
                     func_col: str,
                     out_col_name: str,
                     out_col_empty: Any = 0) -> pd.DataFrame:
    """Filters events to _types and applies func.

    Used to only apply function to relevant types.

    Args:
        events: Events dataframe.
        _types: Types to apply function to.
        func: function to apply.
        func_col: column of events to apply function to.
        out_col_name: Name of column to create.
        out_col_empty: Value for uncalculated rows.

    Returns:
        events: Events dataframe with new column.
    """
    mask = events["type"].isin(_types)
    events_valid = events[mask]
    events[out_col_name] = out_col_empty
    events.loc[mask, out_col_name] = events_valid[func_col].apply(func)
    return events


def groupby_apply_name(events: pd.DataFrame,
                       func: Callable,
                       col_name: str,
                       _type: Optional[str] = None,
                       groupby=["match_id", "team"]):
    events_groupby = events.groupby(groupby)
    out = events_groupby.apply(func, _type=_type)
    out.name = col_name
    return out

# The patter should be: 
# Optional: Calculating something from the features as a column
# Taking a statistical measure of it in a function

# name of func is explanatory
def _pass_is_sideways(pass_angle: float):
    if np.pi / 4 < np.abs(pass_angle) < 3 * np.pi / 4:
        return True
    return False


def _pass_is_forward(pass_angle: float):
    if np.abs(pass_angle) <= np.pi / 4:
        return True
    return False


def _pass_is_backward(pass_angle: float):
    if np.abs(pass_angle) >= 3 * np.pi / 4:
        return True
    return False


def _pass_is_long(pass_distance: float):
    if pass_distance > 60:  # TODO tune
        return True
    return False


def _pass_is_medium(pass_distance: float):
    if 18 < pass_distance <= 60:  # TODO tune
        return True
    return False


def _pass_is_short(pass_distance: float):
    if pass_distance <= 18:  # TODO tune
        return True
    return False


# Aggregation features
# TODO time on ball features
# TODO create tests for features

# TODO: Argument - easier just to create specific functions
# for locations, and combine


def normalised_count(events: pd.DataFrame, _type: str):
    """Filters event_types to type, and divides by tot_events"""
    out = groupby_apply_name(events,
                             _normalised_count,
                             f"{_type}_normalised_count",
                             _type=_type)

def _normalised_count(events: pd.DataFrame, _type: str) -> int:
    """Filters event_types to type, and divides by tot_events"""
    tot_events = events.shape[0]
    mask = events["type"] == _type
    n_events = events[mask].shape[0]
    return n_events / tot_events


def count_events(events: pd.DataFrame, _type: str) -> int:
    """Counts number of events of _type"""
    mask = events["type"] == _type
    n_events = events[mask].shape[0]
    return n_events


def shot_aerial_won_ratio(events: pd.DataFrame) -> float:
    """Filters to shots, then calculates mean of shot_aerial_won"""
    out = groupby_apply_name(events,
                             _shot_aerial_won_ratio,
                             "shot_aerial_won_ratio")
    return out


def _shot_aerial_won_ratio(events: pd.DataFrame, _type: str=None):
    mask = events["type"] == "Shot"
    shot_events = events[mask]
    shot_events.loc[shot_events["shot_aerial_won"].isnull()] = 0
    return shot_events["shot_aerial_won"].mean()


def shot_first_time_ratio(events:pd.DataFrame):
    out = groupby_apply_name(events, _shot_first_time_ratio,
                             "shot_first_time_ratio")
    return out


def _shot_first_time_ratio(events: pd.DataFrame, _type: str=None) -> float:
    """Filters to shots, then calculates mean of shot_first_time"""
    mask = events["type"] == "Shot"
    shot_events = events[mask]
    shot_events.loc[shot_events["shot_first_time"].isnull()] = 0
    return shot_events["shot_first_time"].mean()


def pass_forward_ratio(events: pd.DataFrame):
    if "pass_forwards" not in events.columns:
        events = filter_and_apply(events,
                                  ["Pass"],
                                  _pass_is_forward,
                                  "pass_angle",
                                  "pass_forwards",
                                  )
    out = groupby_apply_name(events,
                             _direction_forward_ratio,
                             "pass_forward_ratio")
    return out


def _direction_forward_ratio(events: pd.DataFrame, _type: str=None) -> float:
    """Filters to passes, then finds mean of pass_forwards"""
    mask = events["type"] == "Pass"
    pass_events = events[mask]
    return pass_events["pass_forwards"].mean()


def pass_backward_ratio(events: pd.DataFrame):
    if "pass_backwards" not in events.columns:
        events = filter_and_apply(events,
                                  ["Pass"],
                                  _pass_is_backward,
                                  "pass_angle",
                                  "pass_backwards",
                                  )
    out = groupby_apply_name(events,
                             _direction_backward_ratio,
                             "pass_backward_ratio")
    return out


def _direction_backward_ratio(events: pd.DataFrame, _type: str=None) -> float:
    """percentage of passes backward."""
    mask = events["type"] == "Pass"
    pass_events = events[mask]
    return pass_events["pass_backwards"].mean()


def pass_sideways_ratio(events: pd.DataFrame):
    if "pass_sideways" not in events.columns:
        events = filter_and_apply(events,
                                  ["Pass"],
                                  _pass_is_sideways,
                                  "pass_angle",
                                  "pass_sideways",
                                  )
    out = groupby_apply_name(events,
                             _direction_sideways_ratio,
                             "pass_sideways_ratio")
    return out


def _direction_sideways_ratio(events: pd.DataFrame, _type: str=None) -> float:
    """percentage of passes sideways"""
    mask = events["type"] == "Pass"
    pass_events = events[mask]
    return pass_events["pass_sideways"].mean()


def high_pass_ratio(events: pd.DataFrame) -> float:
    """Filters to passes and height: High Pass and calculates ratio of high:low"""
    mask = events["type"] == "Pass"
    pass_events = events[mask]
    tot_pass_events = pass_events.shape[0]
    high_pass_events = pass_events[pass_events["height"] == "High Pass"]
    n_high_pass_events = high_pass_events.shape[0]
    return n_high_pass_events / tot_pass_events


def pass_short_ratio(events: pd.DataFrame):
    if "pass_short" not in events.columns:
        events = filter_and_apply(events,
                                  ["Pass"],
                                  _pass_is_short,
                                  "pass_length",
                                  "pass_short")
    out = groupby_apply_name(events,
                             _distance_short_ratio,
                             "pass_short_ratio")
    return out


# TODO configure pass lengths
def _distance_short_ratio(events: pd.DataFrame) -> float:
    """Filters to pass and finds mean of pass_short"""
    
    mask = events["type"] == "Pass"
    pass_events = events[mask]
    return pass_events["pass_short"].mean()


def pass_medium_ratio(events: pd.DataFrame):
    if "pass_medium" not in events.columns:
        events = filter_and_apply(events,
                                  ["Pass"],
                                  _pass_is_medium,
                                  "pass_length",
                                  "pass_medium")
    out = groupby_apply_name(events,
                             _distance_medium_ratio,
                             "pass_medium_ratio")
    return out


def _distance_medium_ratio(events: pd.DataFrame) -> float:
    """Filters to pass and finds mean of pass_medium"""
    
    mask = events["type"] == "Pass"
    pass_events = events[mask]
    return pass_events["pass_medium"].mean()


def pass_long_ratio(events: pd.DataFrame):
    if "pass_long" not in events.columns:
        events = filter_and_apply(events,
                                  ["Pass"],
                                  _pass_is_long,
                                  "pass_length",
                                  "pass_long")
    out = groupby_apply_name(events,
                             _distance_long_ratio,
                             "pass_long_ratio")
    return out


def _distance_long_ratio(events: pd.DataFrame) -> float:
    """Filters to pass and finds mean of pass_long"""
    
    mask = events["type"] == "Pass"
    pass_events = events[mask]
    return pass_events["pass_long"].mean()


def pass_success_rate(events: pd.DataFrame) -> float:
    """Filters to pass and finds pass_outcome == null / tot_passes"""
    mask = events["type"] == "Pass"
    pass_events = events[mask]
    tot_pass_events = pass_events.shape[0]
    null_outcomes = pass_events["pass_outcome"].isnull()
    n_null_outcomes = null_outcomes.sum()
    return n_null_outcomes / tot_pass_events


def pass_cross_ratio(events: pd.DataFrame) -> float:
    """Filters to pass and finds mean of pass_cross"""
    mask = events["type"] == "Pass"
    pass_events = events[mask]
    pass_events.loc[pass_events["pass_cross"].isnull()] = 0
    return pass_events["pass_cross"].mean()


