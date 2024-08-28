"""Designations and column names for stats and points used in my league."""

# CSV column name mapped to number of IDP points the feature is worth.
FANTASY_POINTS = {
  # == Passing ==
  "passing_yards": 0.04,
  "passing_tds": 4,
  "passing_2pt_conversions": 2,
  "interceptions": -2,
  # == Rushing ==
  "rushing_yards": 0.1,
  "rushing_tds": 6,
  "rushing_2pt_conversions": 2,
  # == Receiving ==
  "receptions": 1,
  "receiving_yards": 0.1,
  "receiving_tds": 6,
  "receiving_2pt_conversions": 2,
  # == Kicking ==
  "fg_made_distance": 0.1,
  "pat_made": 1,
  "fg_missed": -1,
  "pat_missed": -1,
  # == Special Teams Player ==
  "special_teams_tds": 6,
  # Note: I hope the special teams version of forced fumble, recovery, solo
  # tackle, are tabulated in the `def` CSV...
  # Misc:
  "receiving_fumbles_lost": -2,
  "receiving_fumbles_lost": -2,
  "sack_fumbles_lost": -2,
  # Note: no "fumble recovery TD" entry
  # == IDP ==
  "def_tds": 6, # a.k.a., IDP TD
  "def_sacks": 4,
  "def_tackles_for_loss": 2,
  # Note: No blocked punt/PAT/FG
  "def_interceptions": 5,
  "def_fumble_recovery_opp": 2,
  "def_fumble_recovery_own": 2,
  "def_fumbles_forced": 2,
  "def_safety": 2,
  "def_tackles_with_assist": 0.75,
  "def_tackles_solo": 1.5,
  "def_pass_defended": 1.5,            
}


# Stats that are worth zero IDP points, but are useful(?) predictors.
PREDICTORS = (
  "air_yards_share",
  "attempts",
  "carries",
  "completions",
  "dakota",
  "fantasy_points",
  "fantasy_points_ppr",
  "games",
  "pacr",
  "passing_air_yards",
  "passing_epa",
  "passing_first_downs",
  "passing_yards_after_catch",
  "racr",
  "receiving_air_yards",
  "receiving_epa",
  "receiving_first_downs",
  "receiving_fumbles",
  "receiving_yards_after_catch",
  "rushing_epa",
  "rushing_first_downs",
  "rushing_fumbles",
  "rushing_fumbles_lost",
  "sack_fumbles",
  "sack_yards",
  "sacks",
  "target_share",
  "targets",
  "wopr",
  "def_fumble_recovery_yards_opp",
  "def_fumble_recovery_yards_own",
  "def_fumbles",
  "def_interception_yards",
  "def_penalty",
  "def_penalty_yards",
  "def_qb_hits",
  "def_sack_yards",
  "def_tackle_assists",
  "def_tackles",
  "def_tackles_for_loss_yards",
  "def_games",  # WARNING!! MUST MAP!! Manually edit CSVs.
  "fg_att",
  "fg_blocked",
  "fg_blocked_distance",
  "fg_long",
  "fg_made",
  "fg_made_0_19",
  "fg_made_20_29",
  "fg_made_30_39",
  "fg_made_40_49",
  "fg_made_50_59",
  "fg_made_60_",
  "fg_missed_0_19",  # Note: this is always zero in 2022-23 season!
  "fg_missed_20_29",
  "fg_missed_30_39",
  "fg_missed_40_49",
  "fg_missed_50_59",
  "fg_missed_60_",
  "fg_missed_distance",
  "fg_pct",
  "kck_games",  # WARNING!! MUST MAP! Manually edit CSVs.
  "gwfg_att",
  "gwfg_blocked",
  "gwfg_made",
  "gwfg_missed",
  "pat_att",
  "pat_blocked",
  "pat_pct",
)


ALL_FEATURES = tuple(sorted(PREDICTORS + tuple(FANTASY_POINTS.keys())))


TEAMS = (
  "ARI", "ATL", "BAL", "BUF", "CAR",  # 0 .. 4
  "CHI", "CIN", "CLE", "DAL", "DEN",  # 5 .. 9
  "DET", "GB", "HOU", "IND", "JAX",  # 10 .. 14
  "KC", "LA", "LAC", "LV", "MIA",  # 15 .. 19
  "MIN", "NE", "NO", "NYG", "NYJ",  # 20 .. 24
  "PHI", "PIT", "SEA", "SF", "TB",  # 25 .. 29
  "TEN", "WAS",  # 30, 31
)

POSITIONS = (
  "C", "CB", "DB", "DE", "DT",  # 0 .. 4
  "FB", "FS", "G", "ILB", "K",  # 5 .. 9
  "LB", "MLB", "NT", "OLB", "P",  # 10 .. 14
  "QB", "RB", "SS", "T", "TE",  # 20 .. 24
  "WR",  # 25
)