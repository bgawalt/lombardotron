# LombardoTron

I got a fantasy football draft coming up Monday Aug 26, 2024, and I need a way
to rank players.  So I will try out some models in SciKit-Learn that map
year K's stats to year K+1's overall performance.

It's an IDP league, so I gotta pay attention to offense and defense.  Which,
look, this is all the same to me, I'm outsourcing all this to some random
forest anyway.

## Python environment

```shell
$ pip3 install -U scikit-learn
```

Results:

```shell
$ pip freeze
joblib==1.4.2
numpy==2.1.0
scikit-learn==1.5.1
scipy==1.14.1
threadpoolctl==3.5.0
```

## Data

Much thanks to NFLVerse for these CSVs:

https://github.com/nflverse/nflverse-data/releases/tag/player_stats

I downloaded a few into a folder called `data/` that I told git to ignore.

CSV columns:

```
SEASON 2023
0 : player_id
1 : player_name
2 : player_display_name
3 : position
4 : position_group
5 : headshot_url
6 : recent_team
7 : season
8 : week
9 : season_type
10 : opponent_team
11 : completions
12 : attempts
13 : passing_yards
14 : passing_tds
15 : interceptions
16 : sacks
17 : sack_yards
18 : sack_fumbles
19 : sack_fumbles_lost
20 : passing_air_yards
21 : passing_yards_after_catch
22 : passing_first_downs
23 : passing_epa
24 : passing_2pt_conversions
25 : pacr
26 : dakota
27 : carries
28 : rushing_yards
29 : rushing_tds
30 : rushing_fumbles
31 : rushing_fumbles_lost
32 : rushing_first_downs
33 : rushing_epa
34 : rushing_2pt_conversions
35 : receptions
36 : targets
37 : receiving_yards
38 : receiving_tds
39 : receiving_fumbles
40 : receiving_fumbles_lost
41 : receiving_air_yards
42 : receiving_yards_after_catch
43 : receiving_first_downs
44 : receiving_epa
45 : receiving_2pt_conversions
46 : racr
47 : target_share
48 : air_yards_share
49 : wopr
50 : special_teams_tds
51 : fantasy_points
52 : fantasy_points_ppr


DEFENSE
0 : season
1 : season_type
2 : player_id
3 : player_name
4 : player_display_name
5 : games
6 : position
7 : position_group
8 : headshot_url
9 : team
10 : def_tackles
11 : def_tackles_solo
12 : def_tackles_with_assist
13 : def_tackle_assists
14 : def_tackles_for_loss
15 : def_tackles_for_loss_yards
16 : def_fumbles_forced
17 : def_sacks
18 : def_sack_yards
19 : def_qb_hits
20 : def_interceptions
21 : def_interception_yards
22 : def_pass_defended
23 : def_tds
24 : def_fumbles
25 : def_fumble_recovery_own
26 : def_fumble_recovery_yards_own
27 : def_fumble_recovery_opp
28 : def_fumble_recovery_yards_opp
29 : def_safety
30 : def_penalty
31 : def_penalty_yards
```

Confirmed this is the same for '22 and for '23.