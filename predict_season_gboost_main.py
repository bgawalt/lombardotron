"""Predict total IDP score earnings of players in 2025 from their 2024 stats.

Produces a ranking of players using a gradient boosting forest.

Usage:

  $ python predict_season_gboost_main.py rankings.csv

"""

import csv
import dataclasses
import hashlib
import sys

import numpy

from sklearn import ensemble # type: ignore

import common
import seasonstats
import weekonestats


FEATURES = (
  tuple("next_week_one_" + f for f in weekonestats.WEEK_ONE_FEATURES) +
  tuple("prev_week_one_" + f for f in weekonestats.WEEK_ONE_FEATURES) +
  tuple("prev_season_" + f for f in seasonstats.SEASON_STAT_FEATURES) +
  tuple("prev_season_off_games_" + t for t in common.TEAMS) + 
  tuple("prev_season_def_games_" + t for t in common.TEAMS) + 
  tuple("prev_season_kck_games_" + t for t in common.TEAMS) + 
  tuple("prev_season_pos_games_" + t for t in common.POSITIONS)
)


@dataclasses.dataclass
class LabelledExamples:
  pids: tuple[str, ...]
  features: numpy.ndarray
  labels: tuple[float, ...]
  weights: tuple[float, ...]

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
  
  def split(
      self,
      salt: str = "",
      fraction: float = 0.8
  ) -> tuple["LabelledExamples", "LabelledExamples"]:
    leftp = []
    leftf = []
    leftl = []
    leftw = []
    rightp = []
    rightf = []
    rightl = []
    rightw = []
    big_number = 1000000
    threshold = fraction * big_number
    plw = zip(self.pids, self.labels, self.weights)
    for i, (pid, label, weight) in enumerate(plw):
      salty_pid = (salt + pid).encode("utf-8")
      hashcode = int(hashlib.sha256(salty_pid).hexdigest(), 16)
      if (hashcode % big_number) <= threshold:
        leftp.append(pid)
        leftf.append(i)
        leftl.append(label)
        leftw.append(weight)
      else:
        rightp.append(pid)
        rightf.append(i)
        rightl.append(label)
        rightw.append(weight)
    return (
      LabelledExamples(
        pids=tuple(leftp),
        features=self.features[leftf, :],
        labels=tuple(leftl),
        weights=tuple(leftw)
      ),
      LabelledExamples(
        pids=tuple(rightp),
        features=self.features[rightf, :],
        labels=tuple(rightl),
        weights=tuple(rightw)
      ),
    )
  

  # TODO: This should just take a sequence of LabelledExamples as a single
  # argument instead of `first`, `second`, `third`....
  @classmethod
  def merge(
    cls,
    first: "LabelledExamples",
    second: "LabelledExamples",
    third: "LabelledExamples",
    weight_decay: float
  ) -> "LabelledExamples":
    merge_pids = first.pids + second.pids + third.pids
    merge_feats = numpy.vstack(
      [first.features, second.features, third.features])
    merge_labels = first.labels + second.labels + third.labels
    merge_weights = (
      first.weights +
      tuple(weight_decay * swi for swi in second.weights) + 
      tuple(weight_decay * weight_decay * twi  for twi in third.weights)
    )
    return LabelledExamples(
      pids=merge_pids,
      features=merge_feats,
      labels=merge_labels,
      weights=merge_weights
    )    


def build_labelled_examples(
    prev_roster: weekonestats.WeekOneLeague,
    prev_season: seasonstats.SeasonStats,
    next_roster: weekonestats.WeekOneLeague,
    next_season: seasonstats.SeasonStats) -> LabelledExamples:
  pids = []
  features = []
  labels = []
  weights = []
  prev_pids = set(prev_season.player_ids)
  nwos = weekonestats.NUM_WEEK_ONE_FEATURES
  for pid in next_season.player_ids:
    if pid not in next_roster.players:
      continue
    pids.append(pid)
    next_season_stats = next_season.get_player_stats(pid)
    labels.append(next_season_stats.idp_score())
    weights.append(next_season_stats.weight())
    vi = numpy.zeros((1, len(FEATURES)), float)
    vi[0, :nwos] = next_roster.players[pid].features()
    if pid in prev_roster.players:
      vi[0, nwos:(2 * nwos)] = prev_roster.players[pid].features()
    if pid in prev_pids:
      prev_season_stats = prev_season.get_player_stats(pid)
      vi[0, (2 * nwos):] = prev_season_stats.features()
    features.append(vi)
  matrix = numpy.vstack(features)
  return LabelledExamples(
    pids=tuple(pids),
    features=matrix,
    labels=tuple(labels),
    weights=tuple(weights)
  )


def build_unlabelled_examples(
    prev_roster: weekonestats.WeekOneLeague,
    prev_season: seasonstats.SeasonStats,
    next_roster: weekonestats.WeekOneLeague,
) -> LabelledExamples:
  pids = []
  features = []
  prev_pids = set(prev_season.player_ids)
  nwos = weekonestats.NUM_WEEK_ONE_FEATURES
  for pid in next_roster.players:
    pids.append(pid)
    vi = numpy.zeros((1, len(FEATURES)), float)
    vi[0, :nwos] = next_roster.players[pid].features()
    if pid in prev_roster.players:
      vi[0, nwos:(2 * nwos)] = (
        prev_roster.players[pid].features())
    if pid in prev_pids:
      prev_season_stats = prev_season.get_player_stats(pid)
      vi[0, (2 * nwos):] = prev_season_stats.features()
    features.append(vi)
  matrix = numpy.vstack(features)
  return LabelledExamples(
    pids=tuple(pids),
    features=matrix,
    labels=tuple(0 for _ in pids),
    weights=tuple(0 for _ in pids),
  )
  

def main():
  s21 = seasonstats.SeasonStats(seasonstats.SEASON_FILES_2021, "REG")
  s22 = seasonstats.SeasonStats(seasonstats.SEASON_FILES_2022, "REG")
  s23 = seasonstats.SeasonStats(seasonstats.SEASON_FILES_2023, "REG")
  s24 = seasonstats.SeasonStats(seasonstats.SEASON_FILES_2024, "REG")
  r21 = weekonestats.WeekOneLeague(weekonestats.ROSTER_FILE_2021)
  r22 = weekonestats.WeekOneLeague(weekonestats.ROSTER_FILE_2022)
  r23 = weekonestats.WeekOneLeague(weekonestats.ROSTER_FILE_2023)
  r24 = weekonestats.WeekOneLeague(weekonestats.ROSTER_FILE_2024)
  r25 = weekonestats.WeekOneLeague(weekonestats.ROSTER_FILE_2025)
  print("Successfully loaded data from 2021 to 2025")

  s24_from_s23 = build_labelled_examples(
    prev_roster=r23, prev_season=s23, next_roster=r24, next_season=s24)
  print(s24_from_s23.features.shape)
  s23_from_s22 = build_labelled_examples(
    prev_roster=r22, prev_season=s22, next_roster=r23, next_season=s23)
  print(s23_from_s22.features.shape)
  s22_from_s21 = build_labelled_examples(
    prev_roster=r21, prev_season=s21, next_roster=r22, next_season=s22)
  print(s22_from_s21.features.shape)

  s25_from_s24 = build_unlabelled_examples(
    prev_roster=r24, prev_season=s24, next_roster=r25)
  print(s25_from_s24.features.shape)

  train = LabelledExamples.merge(s24_from_s23, s23_from_s22, s22_from_s21, 0.9)
  print(f"Train feature matrix shape: {train.features.shape}")

  gbr = ensemble.GradientBoostingRegressor()
  gbr.fit(train.features, train.labels, train.weights)
  print(f"GBR R-squared: {gbr.score(train.features, train.labels, sample_weight=train.weights):0.3f}")
  print(s25_from_s24.features.shape)

  # Save predictions:
  preds = gbr.predict(s25_from_s24.features)
  ranking_fields = [
    "pid",
    "full_name",
    "position",
    "team",
    "predicted_idp",
    "drafted",
    "short_name",
  ]
  with open(sys.argv[1], "wt", newline="") as rankfile:
    writer = csv.DictWriter(rankfile, fieldnames=ranking_fields)
    writer.writeheader()
    pid_pred_pairs = sorted(
      zip(s25_from_s24.pids, preds), key=lambda t: t[1], reverse=True)
    for pid, pred in pid_pred_pairs:
      player = r25.players[pid]
      writer.writerow({
        "pid": pid,
        "full_name": player.name,
        "position": player.position,
        "team": player.team,
        "predicted_idp": f"{pred:0.3f}",
        "drafted": "",
        "short_name": player.short_name,
      })


if __name__ == "__main__":
    main()