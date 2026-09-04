"""Season stats let you load player's season-long stat totals."""

import collections
import csv
import dataclasses
import itertools

from collections.abc import Iterator

import numpy

import common


# CSV column name mapped to number of IDP points the feature is worth.
FANTASY_POINTS = {
  # == Passing ==
  "passing_yards": 0.04,
  "passing_tds": 4,
  "passing_2pt_conversions": 2,
  "def_interceptions": -2,
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
  "fumble_recovery_opp": 2,
  "fumble_recovery_own": 2,
  "def_fumbles_forced": 2,
  "def_safeties": 2,
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
  "fantasy_points",
  "fantasy_points_ppr",
  "games",
  "def_games",  # WARNING!! MUST MAP!! Manually edit CSVs.
  "kck_games",  # WARNING!! MUST MAP! Manually edit CSVs.
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
  "def_sack_yards",
  "def_sacks",
  "target_share",
  "targets",
  "wopr",
  "fumble_recovery_yards_opp",
  "fumble_recovery_yards_own",
  "def_fumbles",
  "def_interception_yards",
  "penalties",
  "penalty_yards",
  "def_qb_hits",
  "def_sack_yards",
  "def_tackle_assists",
  "def_tackles_for_loss_yards",
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
  "gwfg_att",
  "gwfg_blocked",
  "gwfg_made",
  "gwfg_missed",
  "pat_att",
  "pat_blocked",
  "pat_pct",
)

SEASON_STAT_FEATURES = tuple(sorted(PREDICTORS + tuple(FANTASY_POINTS.keys())))

NUM_SEASON_FEATURES = (
  len(SEASON_STAT_FEATURES) +
  (3 * len(common.TEAMS)) +
  len(common.POSITIONS)
)


SEASON_2021 = "./data/stats_player_reg_2021.csv"
SEASON_2022 = "./data/stats_player_reg_2022.csv"
SEASON_2023 = "./data/stats_player_reg_2023.csv"
SEASON_2024 = "./data/stats_player_reg_2024.csv"
SEASON_2025 = "./data/stats_player_reg_2025.csv"

PID_COLUMN = "player_id"
NAME_COLUMN = "player_display_name"
POSITION_COLUMN = "position"


class PlayerSeason:
  """Stats for one player, for one season."""

  def __init__(self, pid: str, name: str):
    self._pid = pid
    self._name = name
    self._positions = collections.defaultdict(float)
    self._teams = collections.defaultdict(float)
    self._stats = collections.defaultdict(float)
  
  @property
  def name(self) -> str:
    return self._name

  def add_row(self, row: dict[str, str]):
    """Add per-team season-long off/def/kick statistics for a player."""
    pos = row[POSITION_COLUMN]
    self._positions[pos] += common.empty_float(row.get("games", "0"))
    # TODO: Bring back OFF, DEF, KCK game count?
    team = row["recent_team"]
    if team in self._teams:
      raise ValueError(f"Multiple insertion, {team}, {self._pid}")
    self._teams[team] = common.empty_float(row["games"])
    for stat in SEASON_STAT_FEATURES:
      if stat not in row:
        continue
      self._stats[stat] += common.empty_float(row[stat])
  
  def roles(self) -> str:
    return "/".join(
      role for role, _ in
      sorted(self._positions.items(), key=lambda t: t[1], reverse=True)
    )    
  
  def idp_score(self) -> float:
    """Points earned by player over the season under my league's IDP rules."""
    return sum(
      pts * self._stats[stat]
      for stat, pts in FANTASY_POINTS.items()
    )
  
  def weight(self) -> float:
    """How much influence to give this player when training an IDP predictor.
    
    Each team in my league has 19 slots, and there's 12 league members, so 228
    drafted players. If you look at the 250th highest IDP score in 2023, that's
    around 117 points. So most of the players I'm interested in, have an IDP
    score above 100 points.

    This function weights each player based on that threshold of 100 IDP points.
    """
    return max(self.idp_score()/100, 1.0)

  def _numeric_features(self) -> Iterator[float]:
    yield from (self._stats.get(stat, 0.0) for stat in SEASON_STAT_FEATURES)
  
  def _team_features(self) -> Iterator[float]:
    yield from (self._teams.get(team, 0.0) for team in common.TEAMS)
  
  def _position_features(self) -> Iterator[float]:
    yield from (self._positions[pos] for pos in common.POSITIONS)
  
  def features(self) -> numpy.ndarray:
    return numpy.fromiter(
      itertools.chain(
        self._numeric_features(),
        self._team_features(),
        self._position_features()
      ),
      float
    ).reshape((1, NUM_SEASON_FEATURES))


class SeasonStats:
  """Stats for the full league of players, for one season."""

  def __init__(self, season_csv: str):
    self._players: dict[str, PlayerSeason] = {}
    with open(season_csv, "rt") as infile:
      for row in csv.DictReader(infile):
        pid = row[PID_COLUMN]
        name = row[NAME_COLUMN]
        if pid not in self._players:
          self._players[pid] = PlayerSeason(pid=pid, name=name)
        self._players[pid].add_row(row)

  @property
  def player_ids(self) -> Iterator[str]:
    """All the player IDs recorded for this season."""
    yield from self._players.keys()
  
  def get_player_stats(self, player_id: str) -> PlayerSeason:
    """Stats for the player for this season, if available."""
    return self._players[player_id]