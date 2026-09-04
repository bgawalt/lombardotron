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
*  def_safety -> def_safeties
*  def_fumble_recovery_{own, opp} -> fumble_recovery_{own, opp}
*  def_fumble_recovery_yards_{own, opp} -> fumble_recovery_yards_{own, opp}
*  sacks -> def_sacks
*  interceptions -> def_interceptions
*  dakota -> obsolete, drop
*  def_penalty -> penalties
*  sack_yards -> def_sack_yards

ok, set em up, knock em down.

I will hold out on refactoring the `def_`, `kck_` etc particulars, and instead
try a regularized quadratic kernel.  If it's not any better than the linear
model, *then* I might go back and manually add interaction terms.

Oh noooooo the results are terrible.  The model thinks it's fit well:

```
Train feature matrix shape: (5824, 163)
OLS R-squared: 0.687
```

but it's ranking the wrong positions way up top

```
pid,full_name,position,team,predicted_idp,drafted,short_name
00-0036224,Jonathan Greenard,LB,PHI,487.755,,J.Greenard
00-0036290,Cole Kmet,TE,CHI,458.863,,C.Kmet
00-0034361,Justin Reid,DB,NO,409.336,,J.Reid
00-0037364,Arron Mosby,LB,GB,389.151,,A.Mosby
00-0040236,Kyle Monangai,RB,CHI,372.806,,K.Monangai
00-0040105,Jayson Jones,DL,TB,362.957,,J.Jones
00-0036281,C.J. Henderson,DB,ATL,362.669,,C.Henderson
00-0036423,Albert Okwuegbunam,TE,LV,362.064,,A.Okwuegbunam
00-0038046,Charlie Kolar,TE,LAC,356.460,,C.Kolar
```

This was a bug due to not pointing at the right matrix for prediction; it's
fixed now.  I should make this more robust.