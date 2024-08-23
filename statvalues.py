OFF = """
headshot_url",
player_display_name",
player_id",
player_name",
position",
position_group",
recent_team",
season",
season_type",
"""

DEF = """
    
headshot_url",
player_display_name",
player_id",
player_name",
position",
position_group",
season",
season_type",
team",
"""

# CSV column name mapped to number of points the feature is worth.
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
    # TODO: Kicking!!!
    # Special Teams Player
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

# Stats that are worth zero points, but are useful(?) predictors.
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
    "def_games",  # WARNING!! MUST MAP!!
)

ALL_FEATURES = tuple(sorted(PREDICTORS + tuple(FANTASY_POINTS.keys())))