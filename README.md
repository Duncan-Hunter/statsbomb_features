# StatsBomb event aggregation
## For creation of playing style models
Creator: Duncan Hunter

#### Installation
To make use of `feature_engineering.py`, the required packages can be installed with:

`$python -m pip install -r requirements.txt`

It's recommended as always to use a fresh environment. If you've done this, then to *use* the notebook in this folder, your environment will have to be installed as a kernel for jupyter:

`$python -m ipykernel install --user --name env-name --display-name "Python (env-name)"`

Then when opened in Jupyter, select the installed environment. Alternatively, if you're using VSCode, it's notebook feature allows you to select venvs without installing them with ipykernel.

#### Usage

At the moment the functionality of the project isn't 100% complete.

To use aggregating functions available:


```python
from statsbombpy import sb
from statsbomb_features import *

# Get some events:
events = sb.events(3775648)
pass_backward_ratio(events)
```

The features aggregated are as follows :
 - counting of events of a type.
 - normalised_count: Number of events of a type / total number of events.
 - pass_direction_forward_ratio: Percentage of passes that are forward (defined as +45 < a > -45 degrees).
 - pass_direction_sideways_ratio: Percentage of passes that are sideways.
 - pass_direction_backward_ratio: Percentage of passes that are backwards (defined as a > 135, a <-135).
 - pass_short_ratio: Percentage of passes that are < 18 yds. Note these distances will be made configurable.
 - pass_medium_ratio: Percentage of passes that are 18 yds < length > 60 yds.
 - pass_long_ratio: Percentage of passes that are > 60 yds.
 - shot_aerial_won_ratio: Percentage of shots that are aerial won.
 - shot_first_time_ratio: Percentage of shots that are first time.
 - pass_success_ratio: Percentage of passes that are succesful.
 - pass_cross_ratio: Percentage of passes that are crosses.

Features are aggregated over matches and teams. In the future the plan is to aggregate over possession sequences as well, or rolling time windows to have more fine grained details. To implement other types of aggregation, it can be easy as it's a matter of choosing which columns to use in a groupby.

#### Locations:
Future work will make features where location can be taken into account.
Aim is to hardcode a function which filters events down to a certain type, then a function works out what you're interested in, e.g. a specific function for passes in vertical thirds.

#### Development/Testing:
Testing is done in test_fe, and is not very well utilised at the moment.
