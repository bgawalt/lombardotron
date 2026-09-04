# Season 2025

Here is my walkthrough of using `player_stats_*_season_{2021 - 2024}.csv` to
rank the players in `roster_weekly_2025.csv` in terms of, "given what we know
about a player's previous NFL seasons before any '25 games are played, how
many fantasy points will the player accrue over the '25 season."

It's like a lab notebook.

## Updates from 2024

I looked at the league rules.  The fantasy points for each stat seem consistent
with last season (2024), as encoded in `seasonstats.FANTASY_POINTS`.

Train-set squared error was reduced by moving from OLS to a gradient-boosting
forest, so, that's what I'm going with this year. 

For next year, I should set up actual train-test split utils to have a more
sensible crossvalidation bakeoff between the half-dozen different models I might
try.