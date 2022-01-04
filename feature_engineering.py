from sklearn.cluster import KMeans
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
import numpy as np
from typing import Dict, Tuple, List, Any, Callable
from argparse import ArgumentParser
from statsbombpy import sb
from constants import FEATURE_FUNCTIONS, AGGREGATE_FUNCTIONS
from yaml import load, SafeLoader
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


def apply_and_name(events_groupby: pd.DataFrame,
                   func: Callable,
                   name: str):
    out_df = events_groupby.apply(func)
    out_df.name = name
    return out_df
    

# TODO rename these horrendous variables
def apply_config(events: pd.DataFrame,
                 config: Dict) -> pd.DataFrame:
    """Applies the configuration to feature engineering."""

    # Is there a good way to get rid of these loops?
    # Iterating over Types of event in features
    join_dfs = []
    # TODO add groupby to config
    events_groupby = events.groupby(["match_id", "team"])
    for item in config["features"]["Aggregated"].items():
        _type, attributes = item
        col_prefix = _type.lower().replace(" ", "_")
        features = attributes["features"]
        # Add location zones
        if "normalised_location_count" in features:
            location_models = attributes["location_models"]
            for model in location_models:
                location_model = config["location_models"][model]
                midpoints, column_prefix = extract_zone_model(location_model)
                events = event_zone(events,
                                    [_type],
                                    midpoints,
                                    column_prefix)
        # Add features if needed then aggregate


        for feature in features:
            if feature in FEATURE_FUNCTIONS.keys():
                func = FEATURE_FUNCTIONS[feature]["func"]
                func_col = FEATURE_FUNCTIONS[feature]["func_col"]
                out_col_name = FEATURE_FUNCTIONS[feature]["out_col_name"]
                out_col_empty = FEATURE_FUNCTIONS[feature]["null_val"]
                events = filter_and_apply(events,
                                          [_type],
                                          func,
                                          func_col,
                                          out_col_name,
                                          out_col_empty
                                          )
            func = AGGREGATE_FUNCTIONS[feature]
            out_df = apply_and_name(events_groupby, func, feature)
            join_dfs.append(out_df)
    df_final = reduce(
        lambda left, right: pd.merge(left, right, on=["match_id", "team"]),
        join_dfs)
    return df_final
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
               col_name: str) -> pd.DataFrame:
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
    events[col_name] = ""
    events.loc[mask, col_name] = events_valid.location.apply(
        zone_finder, args=[zone_midpoints])
    encoder = OneHotEncoder(sparse=False)
    zone_one_hot = encoder.fit_transform(events[col_name].values.reshape(
        -1, 1))
    _ = {
        col_name + "_" + z: col
        for z, col in zip(encoder.categories_[0], zone_one_hot.T) if z != ""
    }
    zone_df = pd.DataFrame(_)
    events = pd.concat([events, zone_df], axis=1)
    events = events.drop(col_name, axis=1)
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


# name of func is explanatory
def pass_is_sideways(pass_angle: float):
    if np.pi / 4 < np.abs(pass_angle) < 3 * np.pi / 4:
        return True
    return False


def pass_is_forward(pass_angle: float):
    if np.abs(pass_angle) <= np.pi / 4:
        return True
    return False


def pass_is_backward(pass_angle: float):
    if np.abs(pass_angle) >= 3 * np.pi / 4:
        return True
    return False


def pass_is_long(pass_distance: float):
    if pass_distance > 60:  # TODO tune
        return True
    return False


def pass_is_medium(pass_distance: float):
    if 18 < pass_distance <= 60:  # TODO tune
        return True
    return False


def pass_is_short(pass_distance: float):
    if pass_distance <= 18:  # TODO tune
        return True
    return False


# Aggregation features
# TODO time on ball features
# TODO create tests for features
def normalised_location_count(
    events: pd.DataFrame,
    _type: str,
    loc_col_names: List[str],
) -> Dict[str, int]:
    """Creates counts of where events are and divides by total number of events.

    Args:
        events: Events dataframe.
        _type: Type to apply to.
        loc_col_names: Names of one-hot encoded zone columns.

    Returns:
        ratios: Dict of zone:percentage of events in.
    """
    mask = events["type"] == _type
    # Pick events of relevant type
    events_valid = events[mask]
    # Aggregate events_valid
    events_location = events_valid[loc_col_names]
    ratio_events_location = events_location.mean(axis=0)
    ratios = {
        f"ratio_{_type}_{col}": v
        for col, v in zip(loc_col_names, ratio_events_location.values)
    }
    return ratios


def normalised_count(events: pd.DataFrame, _type: str) -> int:
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
    mask = events["type"] == "Shot"
    shot_events = events[mask]
    shot_events.loc[shot_events["shot_aerial_won"].isnull()] = 0
    return shot_events["shot_aerial_won"].mean()


def shot_first_time_ratio(events: pd.DataFrame) -> float:
    """Filters to shots, then calculates mean of shot_first_time"""
    mask = events["type"] == "Shot"
    shot_events = events[mask]
    shot_events.loc[shot_events["shot_first_time"].isnull()] = 0
    return shot_events["shot_first_time"].mean()


def direction_forward_ratio(events: pd.DataFrame) -> float:
    """Filters to passes, then finds mean of pass_forwards"""
    assert "pass_forwards" in events.columns
    mask = events["type"] == "Pass"
    pass_events = events[mask]
    return pass_events["pass_forwards"].mean()


def direction_backward_ratio(events: pd.DataFrame) -> float:
    """percentage of passes backward."""
    assert "pass_backwards" in events.columns
    mask = events["type"] == "Pass"
    pass_events = events[mask]
    return pass_events["pass_backwards"].mean()


def direction_sideways_ratio(events: pd.DataFrame) -> float:
    """percentage of passes sideways"""
    assert "pass_sideways" in events.columns
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

# TODO configure pass lengths
def distance_short_ratio(events: pd.DataFrame) -> float:
    """Filters to pass and finds mean of pass_short"""
    mask = events["type"] == "Pass"
    pass_events = events[mask]
    return pass_events["pass_short"].mean()


def distance_medium_ratio(events: pd.DataFrame) -> float:
    """Filters to pass and finds mean of pass_medium"""
    mask = events["type"] == "Pass"
    pass_events = events[mask]
    return pass_events["pass_medium"].mean()


def distance_long_ratio(events: pd.DataFrame) -> float:
    """Filters to pass and finds mean of pass_long"""
    mask = events["type"] == "Pass"
    pass_events = events[mask]
    return pass_events["pass_long"].mean()


def success_rate(events: pd.DataFrame) -> float:
    """Filters to pass and finds pass_outcome == null / tot_passes"""
    mask = events["type"] == "Pass"
    pass_events = events[mask]
    tot_pass_events = pass_events.shape[0]
    null_outcomes = pass_events["pass_outcome"].isnull()
    n_null_outcomes = null_outcomes.sum()
    return n_null_outcomes / tot_pass_events


def cross_ratio(events: pd.DataFrame) -> float:
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

    # Create features like forward passes, location
    matches_events = event_zone(matches_events,
                                ["Pass", "Shot", "Dribble", "Pressure"],
                                HORIZONTAL_MIDPOINTS, "horizontal_zone")
    matches_events = event_zone(matches_events,
                                ["Pass", "Shot", "Dribble", "Pressure"],
                                VERTICAL_MIDPOINTS, "vertical_zone")

    # Pass events
    matches_events = filter_and_apply(matches_events, ["Pass"],
                                      pass_is_sideways, "pass_angle",
                                      "pass_sideways")
    matches_events = filter_and_apply(matches_events, ["Pass"],
                                      pass_is_forward, "pass_angle",
                                      "pass_forwards")
    matches_events = filter_and_apply(matches_events, ["Pass"],
                                      pass_is_backward, "pass_angle",
                                      "pass_backwards")
    matches_events = filter_and_apply(matches_events, ["Pass"], pass_is_long,
                                      "pass_length", "pass_long")
    matches_events = filter_and_apply(matches_events, ["Pass"], pass_is_medium,
                                      "pass_length", "pass_medium")
    matches_events = filter_and_apply(matches_events, ["Pass"], pass_is_short,
                                      "pass_length", "pass_short")

    hor_loc_col_names = [
        f"horizontal_zone_{k}" for k in HORIZONTAL_MIDPOINTS.keys()
    ]
    ver_loc_col_names = [
        f"vertical_zone_{k}" for k in VERTICAL_MIDPOINTS.keys()
    ]

    matches_events_groupby = matches_events.groupby()
    # TODO provide other groupby's e.g. possession number, rolling time

    # Location Aggregations
    pass_horizontal_ratios = matches_events_groupby.apply(
        normalised_location_count,
        _type="Pass",
        loc_col_names=hor_loc_col_names).apply(pd.Series)
    pass_vertical_ratios = matches_events_groupby.apply(
        normalised_location_count,
        _type="Pass",
        loc_col_names=ver_loc_col_names).apply(pd.Series)

    # TODO make useful shot locations model
    # shot_horizontal_ratios = matches_events_groupby.apply(
    #     normalised_location_count,
    #     _type="Shot",
    #     loc_col_names=hor_loc_col_names).apply(pd.Series)
    # shot_vertical_ratios = matches_events_groupby.apply(
    #     normalised_location_count,
    #     _type="Shot",
    # loc_col_names=ver_loc_col_names).apply(pd.Series)

    pressure_horizontal_ratios = matches_events_groupby.apply(
        normalised_location_count,
        _type="Pressure",
        loc_col_names=hor_loc_col_names).apply(pd.Series)
    pressure_vertical_ratios = matches_events_groupby.apply(
        normalised_location_count,
        _type="Pressure",
        loc_col_names=ver_loc_col_names).apply(pd.Series)

    # Count aggregations
    pass_count = matches_events_groupby.apply(count_events, _type="Pass")
    pass_count.name = "pass_count"
    shot_count = matches_events_groupby.apply(count_events, _type="Shot")
    shot_count.name = "shot_count"
    dribble_count = matches_events_groupby.apply(count_events, _type="Dribble")
    dribble_count.name = "dribble_count"
    # TODO replace strings with constants

    # Normalised count aggregations
    pass_normalised_count = matches_events_groupby.apply(normalised_count,
                                                         _type="Pass")
    pass_normalised_count.name = "pass_normalised_count"
    shot_normalised_count = matches_events_groupby.apply(normalised_count,
                                                         _type="Shot")
    shot_normalised_count.name = "shot_normalised_count"
    dribble_normalised_count = matches_events_groupby.apply(normalised_count,
                                                            _type="Dribble")
    dribble_normalised_count.name = "dribble_normalised_count"

    # Extra features
    pass_direction_forward_ratio = matches_events_groupby.apply(
        direction_forward_ratio)
    pass_direction_forward_ratio.name = "pass_direction_forward_ratio"
    pass_direction_sideways_ratio = matches_events_groupby.apply(
        direction_sideways_ratio)
    pass_direction_sideways_ratio.name = "pass_direction_sideways_ratio"
    pass_direction_backward_ratio = matches_events_groupby.apply(
        direction_backward_ratio)
    pass_direction_backward_ratio.name = "pass_direction_backward_ratio"

    pass_short_ratio = matches_events_groupby.apply(distance_short_ratio)
    pass_short_ratio.name = "pass_short_ratio"
    pass_medium_ratio = matches_events_groupby.apply(distance_medium_ratio)
    pass_medium_ratio.name = "pass_medium_ratio"
    pass_long_ratio = matches_events_groupby.apply(distance_long_ratio)
    pass_long_ratio.name = "pass_long_ratio"

    _shot_aerial_won_ratio = matches_events_groupby.apply(
        shot_aerial_won_ratio)
    _shot_aerial_won_ratio.name = "shot_aerial_won_ratio"
    _shot_first_time_ratio = matches_events_groupby.apply(
        shot_first_time_ratio)
    _shot_first_time_ratio.name = "shot_first_time_ratio"

    pass_success_ratio = matches_events_groupby.apply(success_rate)
    pass_success_ratio.name = "pass_success_ratio"
    pass_cross_ratio = matches_events_groupby.apply(cross_ratio)
    pass_cross_ratio.name = "pass_cross_ratio"

    # TODO - don't have this hardcoded
    join_dfs = [
        pass_horizontal_ratios,
        pass_vertical_ratios,
        shot_horizontal_ratios,
        shot_vertical_ratios,
        pressure_horizontal_ratios,
        pressure_vertical_ratios,
        pass_count,
        shot_count,
        dribble_count,
        pass_normalised_count,
        shot_normalised_count,
        dribble_normalised_count,
        pass_direction_forward_ratio,
        pass_direction_sideways_ratio,
        pass_direction_backward_ratio,
        pass_short_ratio,
        pass_medium_ratio,
        pass_long_ratio,
        shot_aerial_won_ratio,
        shot_first_time_ratio,
        pass_success_ratio,
        pass_cross_ratio,
    ]

    df_final = reduce(
        lambda left, right: pd.merge(left, right, on=["match_id", "team"]),
        join_dfs)
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
