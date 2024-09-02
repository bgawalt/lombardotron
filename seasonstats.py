"""Season stats let you load player's season-long stat totals."""

import collections
import csv
import dataclasses
import itertools

from collections.abc import Iterator

import numpy

import common
import statvalues


NUM_SEASON_FEATURES = (
  len(statvalues.ALL_FEATURES) +
  (3 * len(statvalues.TEAMS)) +
  len(statvalues.POSITIONS)
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
    for stat in statvalues.ALL_FEATURES:
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
      for stat, pts in statvalues.FANTASY_POINTS.items()
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
    yield from (self._stats.get(stat, 0.0) for stat in statvalues.ALL_FEATURES)
  
  def _team_features(self) -> Iterator[float]:
    for role_games in (self._off_games, self._def_games, self._kck_games):
      yield from (role_games.get(team, 0.0) for team in statvalues.TEAMS)
  
  def _position_features(self) -> Iterator[float]:
    yield from (self._positions[pos] for pos in statvalues.POSITIONS)
  
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