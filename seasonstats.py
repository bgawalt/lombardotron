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

SEASON_STAT_FEATURES = tuple(sorted(PREDICTORS + tuple(FANTASY_POINTS.keys())))


NUM_SEASON_FEATURES = (
  len(SEASON_STAT_FEATURES) +
  (3 * len(common.TEAMS)) +
  len(common.POSITIONS)
)

@dataclasses.dataclass(frozen=True)
class SeasonFiles:
  """Paths to CSV files describing player stats over an NFL season."""
  offense_csv: str
  defense_csv: str
  kicking_csv: str


SEASON_FILES_2021 = SeasonFiles(
  offense_csv="./data/player_stats_season_2021.csv",
  defense_csv="./data/player_stats_def_season_2021.csv",
  kicking_csv="./data/player_stats_kicking_season_2021.csv",
)
SEASON_FILES_2022 = SeasonFiles(
  offense_csv="./data/player_stats_season_2022.csv",
  defense_csv="./data/player_stats_def_season_2022.csv",
  kicking_csv="./data/player_stats_kicking_season_2022.csv",
)
SEASON_FILES_2023 = SeasonFiles(
  offense_csv="./data/player_stats_season_2023.csv",
  defense_csv="./data/player_stats_def_season_2023.csv",
  kicking_csv="./data/player_stats_kicking_season_2023.csv",
)

PID_COLUMN = "player_id"
NAME_COLUMN = "player_display_name"
POSITION_COLUMN = "position"


class PlayerSeason:
  """Stats for one player, for one season."""

  def __init__(self, pid: str, name: str):
    self._pid = pid
    self._name = name
    self._positions = collections.defaultdict(float)
    # Each of the following are {team: game count}
    self._off_games = {}
    self._def_games = {}
    self._kck_games = {}
    self._stats = collections.defaultdict(float)
  
  @property
  def name(self) -> str:
    return self._name

  def add_row(self, row: dict[str, str]):
    """Add per-team season-long off/def/kick statistics for a player."""
    pos = row[POSITION_COLUMN]
    self._positions[pos] += (
      common.empty_float(row.get("games", "0")) +
      common.empty_float(row.get("def_games", "0")) +
      common.empty_float(row.get("kck_games", "0"))
    )
    team = None
    if "recent_team" in row:
      team = row["recent_team"]
    elif "team" in row:
      team = row["team"]
    else:
      raise ValueError(f"No team for {self._pid}")
    if "games" in row:
      if team in self._off_games:
        raise ValueError(f"Multiple insertion, offense, {team}, {self._pid}")
      self._off_games[team] = common.empty_float(row["games"])
    elif "def_games" in row:
      if team in self._def_games:
        raise ValueError(f"Multiple insertion, defense, {team}, {self._pid}")
      self._def_games[team] = common.empty_float(row["def_games"])
    elif "kck_games" in row:
      if team in self._kck_games:
        raise ValueError(f"Multiple insertion, kicking, {team}, {self._pid}")
      self._kck_games[team] = common.empty_float(row["kck_games"])
    else:
      raise ValueError(f"No games count for {self._pid}")
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
    for role_games in (self._off_games, self._def_games, self._kck_games):
      yield from (role_games.get(team, 0.0) for team in common.TEAMS)
  
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

  def __init__(self, season: SeasonFiles, season_type: str):
    self._players: dict[str, PlayerSeason] = {}
    self._add_file(filename=season.offense_csv, season_type=season_type)
    self._add_file(filename=season.defense_csv, season_type=season_type)
    self._add_file(filename=season.kicking_csv, season_type=season_type)

  def _add_file(self, filename: str, season_type: str):
    with open(filename, "rt") as infile:
      for row in csv.DictReader(infile):
        if row.get("season_type") != season_type:
          continue
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