from statsbomb_features import *

def test_zone_finder():
    zone_midpoints = {1: (30, 40),
                    2: (90, 40)}
    test_point_1 = [59, 40]
    test_point_2 = [61, 40]
    test_point_3 = [15, 20]
    test_point_4 = [79, 20]

    zone_model = create_zone_model(zone_midpoints)
    assert zone_finder(test_point_1, zone_model, zone_midpoints ) == 1
    assert zone_finder(test_point_2, zone_model, zone_midpoints) == 2

    vertical_tri_zone_midpoints = {1: (30, 40),
                        2: (60, 40),
                        3: (90, 40)}
    zone_model = create_zone_model(vertical_tri_zone_midpoints)
    assert zone_finder(test_point_1, zone_model, vertical_tri_zone_midpoints) == 2
    assert zone_finder(test_point_3, zone_model, vertical_tri_zone_midpoints) == 1
    assert zone_finder(test_point_4, zone_model, vertical_tri_zone_midpoints) == 3


def test_load_and_filter():
    unfiltered_r, unfiltered_c = 131, 20
    unfiltered_df = load_and_filter(37, 90)
    assert unfiltered_df.shape[0] == unfiltered_r
    assert unfiltered_df.shape[1] == unfiltered_c
    filtered_r, filtered_c = 22, 20
    filtered_df = load_and_filter(37, 90, "Arsenal WFC")
    assert filtered_df.shape[0] == filtered_r
    try:
        error_df = load_and_filter(37, 90, "Arsneal WFC")
    except ValueError as e:
        pass
    except:
        raise AssertionError("Didn't identify value Error for mis-spell.")


def test_pass_direction():
    test_match = sb.events(3775648)
    test_pass_side = test_match.loc[8]
    """
    period                          1
    minute                          0
    team                  Arsenal WFC
    possession                      2
    type                         Pass
    pass_type                     NaN
    location             [42.9, 41.1]
    pass_end_location    [40.3, 22.7]
    pass_angle              -1.711171
    player                  Lia Wälti
    timestamp            00:00:02.638
    Name: 8, dtype: object
    """
    assert _pass_is_forward(test_pass_side["pass_angle"]) == 0
    assert _pass_is_backward(test_pass_side["pass_angle"]) == 0
    assert _pass_is_sideways(test_pass_side["pass_angle"]) == 1
    test_pass_back = test_match.loc[7]
    assert _pass_is_forward(test_pass_back["pass_angle"]) == 0
    assert _pass_is_backward(test_pass_back["pass_angle"]) == 1
    assert _pass_is_sideways(test_pass_back["pass_angle"]) == 0
    test_pass_forward = test_match.loc[9]
    assert _pass_is_forward(test_pass_forward["pass_angle"]) == 1
    assert _pass_is_backward(test_pass_forward["pass_angle"]) == 0
    assert _pass_is_sideways(test_pass_forward["pass_angle"]) == 0