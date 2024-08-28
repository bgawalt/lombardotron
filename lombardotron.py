import collections
import csv
import itertools

from collections.abc import Iterator

import numpy

from sklearn import model_selection # type: ignore
from sklearn import svm # type: ignore

import statvalues


_DEF_2022 = "./data/player_stats_def_season_2022.csv"
_DEF_2023 = "./data/player_stats_def_season_2023.csv"
_KCK_2022 = "./data/player_stats_kicking_season_2022.csv"
_KCK_2023 = "./data/player_stats_kicking_season_2023.csv"
_OFF_2022 = "./data/player_stats_season_2022.csv"
_OFF_2023 = "./data/player_stats_season_2023.csv"

_SEASON_2022 = (_OFF_2022, _DEF_2022, _KCK_2022)
_SEASON_2023 = (_OFF_2023, _DEF_2023, _KCK_2023)

# TODO: move REG, POST, REG+POST to statvalues enum

_PID_COLUMN = "player_id"
_NAME_COLUMN = "player_display_name"
_POSITION_COLUMN = "position"
_SEASON_TYPE_COLUMN = "season_type"


def empty_float(s: str) -> float:
  """Parse ASCII to a float, and empty strings count as zero."""
  if not s:
    return 0
  return float(s)


class PlayerStats:
  """Stats for one player, for one season."""

  def __init__(self, pid: str, name: str):
    self._pid = pid
    self._name = name
    self._positions = collections.defaultdict(int)
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
    pos = row[_POSITION_COLUMN]
    self._positions[pos] += 1
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
  
  def features(self) -> numpy.ndarray:
    return numpy.fromiter(
      itertools.chain(self._numeric_features(), self._team_features(),),
      float
    )


class SeasonStats:
  """Stats for the full league of players, for one season."""

  def __init__(self, season: tuple[str, str, str], season_type: str):
    self._players: dict[str, PlayerStats] = {}
    for filename in season:
      self._add_file(filename=filename, season_type=season_type)

  def _add_file(self, filename: str, season_type: str):
    with open(filename, "rt") as infile:
      for i, row in enumerate(csv.DictReader(infile)):
        if row.get("season_type") != season_type:
          continue
        pid = row[_PID_COLUMN]
        name = row[_NAME_COLUMN]
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


def main():
  s22 = SeasonStats(_SEASON_2022, "REG")
  s23 = SeasonStats(_SEASON_2023, "REG")

  s22_pids = set(s22.player_ids)
  both_pids = tuple(pid for pid in s23.player_ids if pid in s22_pids)
  num_rows = len(both_pids)
  num_cols = len(statvalues.ALL_FEATURES) + (3 * len(statvalues.TEAMS))

  features = numpy.zeros((num_rows, num_cols), float)
  labels = numpy.zeros((num_rows,), float)
  weights = []

  for i, pid in enumerate(both_pids):
    s22_pstats = s22.get_player_stats(pid)
    s23_pstats = s23.get_player_stats(pid)
    features[i, :] = s22_pstats.features()
    labels[i] = s23_pstats.idp_score()
    weights.append(s23_pstats.weight())

  params = {"C": [1, 3, 10, 30, 100, 300, 1000]}
  base_svr = svm.SVR(kernel="rbf", gamma="scale", epsilon=5)
  
  k = 7
  inner_cv = model_selection.KFold(
    n_splits=k, shuffle=True, random_state=8675309)
  
  gscv = model_selection.GridSearchCV(
    estimator=base_svr, param_grid=params, cv=inner_cv)
  gscv.fit(features, labels, sample_weight=weights)
  best_c = gscv.best_params_["C"]
  
  svr = svm.SVR(kernel="rbf", C=best_c, gamma="scale", epsilon=5)
  svr.fit(features, labels, sample_weight=weights)
  preds = svr.predict(features)
  print("pid\tname\tactual_idp_23\tpred_idp_22\tpred_svr_teams")
  for pid, pred, actual in zip(both_pids, preds, labels):
    s23_pstats = s23.get_player_stats(pid)
    s22_idp = s22.get_player_stats(pid).idp_score()
    print(
      f"{pid}\t{s23_pstats.name}\t{actual:0.1f}\t{s22_idp:0.1f}\t{pred:0.1f}")
    
  print("\n", features.shape, len(labels))
  print(gscv.cv_results_["mean_test_score"])
  print(gscv.cv_results_["rank_test_score"])
  print(gscv.best_score_, gscv.best_index_)
  print(gscv.best_params_)


if __name__ == "__main__":
    main()