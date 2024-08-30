import collections
import csv
import dataclasses
import itertools

from collections.abc import Iterator
from datetime import datetime

import numpy

from sklearn import model_selection # type: ignore
from sklearn import svm # type: ignore

import statvalues


@dataclasses.dataclass(frozen=True)
class SeasonFiles:
  """Paths to CSV files describing player stats over an NFL season."""
  offense_csv: str
  defense_csv: str
  kicking_csv: str


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

ROSTER_FILE_2022 = "./data/roster_weekly_2022.csv"
ROSTER_FILE_2023 = "./data/roster_weekly_2023.csv"
ROSTER_FILE_2024 = "./data/roster_weekly_2024.csv"

NUM_ROSTER_FEATURES = 8
NUM_SEASON_FEATURES = (
  len(statvalues.ALL_FEATURES) +
  (3 * len(statvalues.TEAMS)) +
  len(statvalues.POSITIONS)
)

NUM_FEATURES = (2 * NUM_ROSTER_FEATURES) + NUM_SEASON_FEATURES

PID_COLUMN = "player_id"
NAME_COLUMN = "player_display_name"
POSITION_COLUMN = "position"


def empty_float(s: str, default: float = 0.0) -> float:
  """Parse ASCII to a float, and empty strings count as zero."""
  if not s:
    return default
  return float(s)


class PlayerStats:
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
      empty_float(row.get("games", "0")) +
      empty_float(row.get("def_games", "0")) +
      empty_float(row.get("kck_games", "0"))
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
      self._off_games[team] = empty_float(row["games"])
    elif "def_games" in row:
      if team in self._def_games:
        raise ValueError(f"Multiple insertion, defense, {team}, {self._pid}")
      self._def_games[team] = empty_float(row["def_games"])
    elif "kck_games" in row:
      if team in self._kck_games:
        raise ValueError(f"Multiple insertion, kicking, {team}, {self._pid}")
      self._kck_games[team] = empty_float(row["kck_games"])
    else:
      raise ValueError(f"No games count for {self._pid}")
    for stat in statvalues.ALL_FEATURES:
      if stat not in row:
        continue
      self._stats[stat] += empty_float(row[stat])
  
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
    self._players: dict[str, PlayerStats] = {}
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
          self._players[pid] = PlayerStats(pid=pid, name=name)
        self._players[pid].add_row(row)

  @property
  def player_ids(self) -> Iterator[str]:
    """All the player IDs recorded for this season."""
    yield from self._players.keys()
  
  def get_player_stats(self, player_id: str) -> PlayerStats:
    """Stats for the player for this season, if available."""
    return self._players[player_id]


@dataclasses.dataclass(frozen=True)
class WeekOnePlayer:
  """Details about a player just before a season's Week 1 kickoff."""
  pid: str
  active: bool
  age: float
  height: float
  weight: float
  years_exp: float
  entry_age: float
  rookie_age: float
  draft_number: float

  def features(self) -> numpy.ndarray:
    return numpy.array([
      1.0 if self.active else 0.0,  # 0
      self.age, self.height, self.weight,  # 1, 2, 3
      self.years_exp, self.entry_age, self.rookie_age,  # 4, 5, 6
      self.draft_number,  # 7
    ], 
    float).reshape((1, NUM_ROSTER_FEATURES))

  @classmethod
  def from_row(cls, row: dict[str, str]) -> 'WeekOnePlayer | None':
    now = datetime.now()
    pid = row["gsis_id"]
    if not pid:
      return None
    if int(row["week"]) != 1:
      return None
    if not row["birth_date"]:
      return None
    birth_date = datetime.strptime(row["birth_date"], "%Y-%m-%d")
    age = (now - birth_date).days / 365
    entry_date = datetime(year=int(row["entry_year"]), month=9, day=1)
    entry_age = (now - entry_date).days / 365
    rook_date = datetime(year=int(row["rookie_year"]), month=9, day=1)
    rook_age = (now - rook_date).days / 365
    "entry_year,rookie_year,draft_club,draft_number"
    return WeekOnePlayer(
      pid=pid,
      active=(row["status"] == "ACT"),
      age=age,
      height=float(row["height"]),
      weight=float(row["weight"]),
      years_exp=float(row["years_exp"]),
      entry_age=entry_age,
      rookie_age=rook_age,
      draft_number=empty_float(row["draft_number"], default=400)
    )


class WeekOneLeague:
  """Details about all players just before a season's Week 1 kickoff."""

  def __init__(self, roster_csv_filename: str):
    self.players: dict[str, WeekOnePlayer] = {}
    with open(roster_csv_filename, "rt") as infile:
      for row in csv.DictReader(infile):
        w1player = WeekOnePlayer.from_row(row)
        if w1player is None:
          continue
        self.players[w1player.pid] = w1player


@dataclasses.dataclass
class LabelledExamples:
  pids: tuple[str]
  features: numpy.ndarray
  labels: tuple[float]
  weights: tuple[float]

  def __post_init__(self):
    if len(self.pids) != len(self.labels):
      raise ValueError(f"len(pids) is {len(self.pids)} "
                       f"but len(labels) is {len(self.labels)}")
    if len(self.pids) != len(self.weights):
      raise ValueError(f"len(pids) is {len(self.pids)} "
                       f"but len(weights) is {len(self.weights)}")
    if len(self.pids) != self.features.shape[0]:
      raise ValueError(f"len(pids) is {len(self.pids)} "
                       f"but len(labels) is {len(self.labels)}")
    


def build_labelled_examples(
    prev_roster: WeekOneLeague,
    prev_season: SeasonStats,
    next_roster: WeekOneLeague,
    next_season: SeasonStats) -> LabelledExamples:
  pids = []
  features = []
  labels = []
  weights = []
  prev_pids = set(prev_season.player_ids)
  for pid in next_season.player_ids:
    if pid not in next_roster.players:
      continue
    pids.append(pid)
    next_season_stats = next_season.get_player_stats(pid)
    labels.append(next_season_stats.idp_score())
    weights.append(next_season_stats.weight())
    vi = numpy.zeros((1, NUM_FEATURES), float)
    vi[0, :NUM_ROSTER_FEATURES] = next_roster.players[pid].features()
    if pid in prev_roster.players:
      vi[0, NUM_ROSTER_FEATURES:(2 * NUM_ROSTER_FEATURES)] = (
        prev_roster.players[pid].features())
    if pid in prev_pids:
      prev_season_stats = prev_season.get_player_stats(pid)
      vi[0, (2 * NUM_ROSTER_FEATURES):] = prev_season_stats.features()
    features.append(vi)
  return LabelledExamples(
    pids=tuple(pids),
    features=numpy.vstack(features),
    labels=tuple(labels),
    weights=tuple(weights)
  )


def main():
  s22 = SeasonStats(SEASON_FILES_2022, "REG")
  s23 = SeasonStats(SEASON_FILES_2023, "REG")
  r22 = WeekOneLeague(ROSTER_FILE_2022)
  r23 = WeekOneLeague(ROSTER_FILE_2023)

  training_data = build_labelled_examples(
    prev_roster=r22, prev_season=s22, next_roster=r23, next_season=s23)


  params = {"C": [1, 3, 10, 30, 100, 300, 1000]}
  base_svr = svm.SVR(kernel="rbf", gamma="scale", epsilon=5)
  
  k = 7
  inner_cv = model_selection.KFold(
    n_splits=k, shuffle=True, random_state=8675309)
  
  gscv = model_selection.GridSearchCV(
    estimator=base_svr, param_grid=params, cv=inner_cv)
  gscv.fit(
    training_data.features,
    training_data.labels,
    sample_weight=training_data.weights
  )
  best_c = gscv.best_params_["C"]
  
  svr = svm.SVR(kernel="rbf", C=best_c, gamma="scale", epsilon=5)
  svr.fit(
    training_data.features,
    training_data.labels,
    sample_weight=training_data.weights
  )
  preds = svr.predict(training_data.features)
  print("pid\tname\tactual_idp_23\tpred_idp_22\tpred_svr_teams")
  for pid, pred, actual in zip(training_data.pids, preds, training_data.labels):
    s23_pstats = s23.get_player_stats(pid)
    try:
      s22_idp = s22.get_player_stats(pid).idp_score()
    except:
      s22_idp = 0
    print(
      f"{pid}\t{s23_pstats.name}\t{actual:0.1f}\t{s22_idp:0.1f}\t{pred:0.1f}")
    
  print("\n", training_data.features.shape, len(training_data.labels))
  print(gscv.cv_results_["mean_test_score"])
  print(gscv.cv_results_["rank_test_score"])
  print(gscv.best_score_, gscv.best_index_)
  print(gscv.best_params_)


if __name__ == "__main__":
    main()