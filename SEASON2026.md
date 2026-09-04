# Season 2025-2026

Most important thing: when I checked the README's 

https://github.com/nflverse/nflverse-data/releases/tag/player_stats

link, it said

`DEPRECATED 2025-08-01: USE stats_player OR stats_team INSTEAD`

so first thing is to see if those stats files are going to match up with
the fields expected by `seasonstats.PREDICTORS` et al.

Here's the check of "does every field in the csv show up in `PREDICTORS` and
vice versa.  Here's what's missing in the "i expect these fields" direction:

```
PREDICTORS had kck_games but csv didnt
PREDICTORS had def_games but csv didnt
PREDICTORS had def_penalty_yards but csv didnt
PREDICTORS had def_tackles but csv didnt
PREDICTORS had def_fumble_recovery_yards_own but csv didnt
PREDICTORS had sacks but csv didnt
PREDICTORS had def_safety but csv didnt
PREDICTORS had def_fumble_recovery_yards_opp but csv didnt
PREDICTORS had interceptions but csv didnt
PREDICTORS had dakota but csv didnt
PREDICTORS had def_penalty but csv didnt
PREDICTORS had def_fumble_recovery_own but csv didnt
PREDICTORS had sack_yards but csv didnt
PREDICTORS had def_fumble_recovery_opp but csv didnt
```

The big upshot: no more splitting players into `def` and `kck` sub-CSVs. 
They're all in one summary file.  That's better in an absolute sense

Patching those holes:

*  kck_games, def_games: obsolete, remove, manually recalcuate
*  def_penalty_yards: manually recalculate
*  def_tackles -> split into _solo and _with_assist (and _assists)
*  sack_yards -> def_sack_yards
*  def_fumble_recovery_{own, opp} -> fumble_recovery_{own, opp}
*  def_fumble_recovery_yards_{own, opp} -> fumble_recovery_yards_{own, opp}
*  sacks -> def_sacks
*  interceptions -> def_interceptions
*  dakota -> obsolete, drop
*  def_penalty -> penalties
*  sack_yards -> def_sack_yards

ok, set em up, knock em down.