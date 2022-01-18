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


def extract_zone_model(location_config: Dict):
    midpoints = location_config["midpoints"]
    column_prefix = location_config["column_prefix"]
    return midpoints, column_prefix


def load_config(config_path: str) -> Dict:
    with open("./config.yaml", "r") as myYaml:
        config = load(myYaml, Loader=SafeLoader)
    return config


    

# TODO rename these horrendous variables
# def apply_config(events: pd.DataFrame,
#                  config: Dict) -> pd.DataFrame:
#     """Applies the configuration to feature engineering."""

#     # Is there a good way to get rid of these loops?
#     # Iterating over Types of event in features
#     join_dfs = []
#     # TODO add groupby to config
#     events_groupby = events.groupby(["match_id", "team"])
#     for item in config["features"]["Aggregated"].items():
#         _type, attributes = item
#         col_prefix = _type.lower().replace(" ", "_")
#         features = attributes["features"]
#         # Add location zones
#         if "normalised_location_count" in features:
#             location_models = attributes["location_models"]
#             for model in location_models:
#                 location_model = config["location_models"][model]
#                 midpoints, column_prefix = extract_zone_model(location_model)
#                 events = event_zone(events,
#                                     [_type],
#                                     midpoints,
#                                     column_prefix)
#         # Add features if needed then aggregate


#         for feature in features:
#             if feature in CONSTANTS.FEATURE_FUNCTIONS.keys():
#                 func = CONSTANTS.FEATURE_FUNCTIONS[feature]["func"]
#             func = CONSTANTS.AGGREGATE_FUNCTIONS[feature]
#             # TODO passing loc col names to func, in general args
#             out_df = apply_and_name(events_groupby, func, _type, feature)
#             join_dfs.append(out_df)

#     df_final = reduce(
#         lambda left, right: pd.merge(left, right, on=["match_id", "team"]),
#         join_dfs)
#     return df_final
            # Add feature if needed
            # calculate feature


# Features that create columns
def zone_finder(
    point: List[float],
    zone_midpoints: Dict[Any, Tuple[int]],
) -> Any:
    """Finds which zone a point is in using Kmeans.

    Args:
        point: location on pitch as [x, y]
        zone_midpoints: Dict of label : midpoint describing zone model.

    Returns:
        zone: Label of zone."""

    point = np.array(point)
    centres = np.array(list(zone_midpoints.values()))
    labels = list(zone_midpoints.keys())
    distances = np.linalg.norm(point - centres, axis=1)
    zone = labels[np.argmin(distances)]
    return zone


def event_zone(events: pd.DataFrame, _types: List[str],
               zone_midpoints: Dict[Any, Tuple[int]],
               location_model_name: str) -> pd.DataFrame:
    """Transforms column of events to one-hot encoded data.

    Args:
        events: events dataframe.
        _types: Types of event to find locations for.
        zone_midpoints: Dict of zones to use in encoding.
        col_name: prefix for output column names.
    Returns:
        events: events dataframe with one-hot encoded features.
    """
    mask = events["type"].isin(_types)
    events_valid = events[mask]
    events[location_model_name] = ""
    events.loc[mask, location_model_name] = events_valid.location.apply(
        zone_finder, args=[zone_midpoints])
    encoder = OneHotEncoder(sparse=False)
    zone_one_hot = encoder.fit_transform(events[location_model_name].values.reshape(
        -1, 1))
    _ = {
        location_model_name + "_" + z: col
        for z, col in zip(encoder.categories_[0], zone_one_hot.T) if z != ""
    }
    zone_df = pd.DataFrame(_)
    events = pd.concat([events, zone_df], axis=1)
    events = events.drop(location_model_name, axis=1)
    return events


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


# TODO use this information
def tactics():
    pass


def main(args):
    matches = load_and_filter(args.competition_id, args.season_id,
                              args.team_name)
    # TODO get config working
    # with open("./features_match.yaml", "r") as file:
    #     config = load(file, Loader=SafeLoader)
    #     attack_config = config["attack"]

    # TODO function for single match (filter to match_id)
    # TODO function for single team (filter to team_id)
    # TODO save time by loading from file rather than downloading

    # Apply seems to be using parallel processing (>95% cpu usage)
    start = time()
    events = matches.head(2).match_id.apply(sb.events)
    events = pd.concat(events.values)
    print("Finished downloading events")
    # Load events of all games into big dataframe
    matches_events = pd.merge(left=matches,
                              right=events,
                              on="match_id",
                              how="inner")
    config_path = os.path.join(args.config_dir, args.config_name)
    print(f"{config_path=}")
    config = load_config(config_path)
    df_final = apply_config(matches_events, config)
    # TODO join scores
    end = time()
    print("Time taken: ", end - start)
    os.makedirs(args.out_dir, exist_ok=True)
    df_final.to_csv(f"{args.out_dir}/{args.out_name}")


if __name__ == "__main__":
    parser = ArgumentParser()
    # Add args
    parser.add_argument("--team_name",
                        type=str,
                        help="Calculate features for a team",
                        required=False)
    parser.add_argument("--competition_id",
                        type=int,
                        help="Which competition is it",
                        default=37)
    parser.add_argument("--season_id",
                        type=int,
                        help="Calculate features for a season",
                        default=90)
    parser.add_argument("--config_dir",
                        type=str,
                        help="Location of config file",
                        default="./",
                        required=False)
    parser.add_argument("--config_name",
                        type=str,
                        help="Name of config file",
                        default="config.yaml")
    # parser.add_argument("--config_file", type=str, help="Path to config file.", default="./config.yaml")
    parser.add_argument("--out_dir",
                        type=str,
                        help="Location to save output file",
                        default="./output")
    parser.add_argument("--out_name",
                        type=str,
                        help="Name of output file",
                        default="aggregated_features.csv")
    args = parser.parse_args()

    main(args)
