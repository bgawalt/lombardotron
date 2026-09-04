"""Predict total IDP score earnings of players in 2025 from their 2024 stats.

Produces a ranking of players, as well as a ridge regression model, and saves
both in the CSVs at the given locations.

Usage:

  $ python predict_season_main.py rankings.csv ridge_coefs.csv

"""

import csv
import dataclasses
import hashlib
import sys

import numpy

from sklearn import linear_model
from sklearn import ensemble

import common
import seasonstats
import weekonestats


FEATURES = (
  tuple("next_week_one_" + f for f in weekonestats.WEEK_ONE_FEATURES) +
  tuple("prev_week_one_" + f for f in weekonestats.WEEK_ONE_FEATURES) +
  tuple("prev_season_" + f for f in seasonstats.SEASON_STAT_FEATURES) +
  tuple("prev_season_teams" + t for t in common.TEAMS) +  
  tuple("prev_season_pos" + t for t in common.POSITIONS)
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
  

def ridge_param_search(train: LabelledExamples) -> float:
  """Steadily narrow a range of log-spaced alphas searched by RidgeCV."""
  lo, hi = (0.01, 100_000_000)  # Powers of ten
  while hi > (1.1 * lo):
    alphas=numpy.logspace(numpy.log10(lo), numpy.log10(hi), num=10)
    rdg = linear_model.RidgeCV(alphas=alphas)
    rdg.fit(train.features, train.labels, train.weights)
    best_idx = list(alphas).index(rdg.alpha_)
    print(
      f"For range [{lo:0.1f}, {hi:0.1f}, best alpha is {rdg.alpha_:0.1f} "
      + f"with score {rdg.best_score_:.5f})"
    )
    lo = alphas[best_idx - 1] if best_idx != 0 else alphas[best_idx]
    hi = alphas[best_idx + 1] if best_idx != len(alphas) - 1 else alphas[best_idx]
  alphas=numpy.logspace(numpy.log10(lo), numpy.log10(hi), num=10)
  rdg = linear_model.RidgeCV(alphas=alphas)
  rdg.fit(train.features, train.labels, train.weights)
  return rdg.alpha_ # type: ignore


def main():
  s21 = seasonstats.SeasonStats(seasonstats.SEASON_2021)
  s22 = seasonstats.SeasonStats(seasonstats.SEASON_2022)
  s23 = seasonstats.SeasonStats(seasonstats.SEASON_2023)
  s24 = seasonstats.SeasonStats(seasonstats.SEASON_2024)
  s25 = seasonstats.SeasonStats(seasonstats.SEASON_2025)
  r21 = weekonestats.WeekOneLeague(weekonestats.ROSTER_FILE_2021)
  r22 = weekonestats.WeekOneLeague(weekonestats.ROSTER_FILE_2022)
  r23 = weekonestats.WeekOneLeague(weekonestats.ROSTER_FILE_2023)
  r24 = weekonestats.WeekOneLeague(weekonestats.ROSTER_FILE_2024)
  r25 = weekonestats.WeekOneLeague(weekonestats.ROSTER_FILE_2025)
  r26 = weekonestats.WeekOneLeague(weekonestats.ROSTER_FILE_2026)
  print("Successfully loaded data from 2021 to 2026")


  print(seasonstats.NUM_SEASON_FEATURES)
  s22_from_s21 = build_labelled_examples(
    prev_roster=r21, prev_season=s21, next_roster=r22, next_season=s22)
  print(s22_from_s21.features.shape)
  s23_from_s22 = build_labelled_examples(
    prev_roster=r22, prev_season=s22, next_roster=r23, next_season=s23)
  print(s23_from_s22.features.shape)
  s24_from_s23 = build_labelled_examples(
      prev_roster=r23, prev_season=s23, next_roster=r24, next_season=s24)
  print(s24_from_s23.features.shape)
  s25_from_s24 = build_labelled_examples(
      prev_roster=r25, prev_season=s24, next_roster=r24, next_season=s24)
  print(s24_from_s23.features.shape)

  s26_from_s25 = build_unlabelled_examples(
    prev_roster=r25, prev_season=s25, next_roster=r26)
  print(s26_from_s25.features.shape)

  train = LabelledExamples.merge(s24_from_s23, s23_from_s22, s22_from_s21, 0.9)
  print(f"Train feature matrix shape: {train.features.shape}")

  # best_alpha = ridge_param_search(train)
  # rdg = linear_model.Ridge(alpha=best_alpha)
  rdg = linear_model.LinearRegression()
  rdg.fit(train.features, train.labels, train.weights)
  print(f"OLS R-squared: {rdg.score(train.features, train.labels, sample_weight=train.weights):0.3f}")

  gbr = ensemble.GradientBoostingRegressor()
  gbr.fit(train.features, train.labels, train.weights)
  print(f"GBR R-squared: {gbr.score(train.features, train.labels, sample_weight=train.weights):0.3f}")

  print(s26_from_s25.features.shape)
  print(rdg.coef_.shape)

  # Save predictions:
  preds = gbr.predict(s26_from_s25.features)
  pid_preds ={pid: pred for pid, pred in zip(s26_from_s25.pids, preds)}

  # What is each player's delta, above the fifth-best player of the same
  # position ranked behind them?
  d5s = {}
  pos_preds = {}
  for pid, pred in pid_preds.items():
    player = r26.players[pid]
    pos = player.position
    if pos not in pos_preds:
      pos_preds[pos] = []
    pos_preds[pos].append((pid, pred))
  for pos, pos_pidpreds in pos_preds.items():
    spreds = sorted(list(pos_pidpreds), key=lambda t: t[1], reverse=True)
    for i, (pid, pred) in enumerate(spreds):
      if i + 5 < len(spreds):
        next_option = spreds[i + 5][1]
      else:
        next_option = 0
      d5s[pid] = pred - next_option
  d5_spids = [
    (pid, d5) for pid, d5 in
    sorted(d5s.items(), key=lambda t: t[1], reverse=True)
  ]

  ranking_fields = [
    "pid",
    "full_name",
    "position",
    "team",
    "predicted_idp",
    "delta5",
    "drafted",
    "short_name",
  ]
  with open(sys.argv[1], "wt", newline="") as rankfile:
    writer = csv.DictWriter(rankfile, fieldnames=ranking_fields)
    writer.writeheader()
    for pid, d5 in d5_spids:
      player = r26.players[pid]
      pred = pid_preds[pid]
      writer.writerow({
        "pid": pid,
        "full_name": player.name,
        "position": player.position,
        "team": player.team,
        "predicted_idp": f"{pred:0.3f}",
        "delta5": f"{d5:0.3f}",
        "drafted": "",
        "short_name": player.short_name,
      })


if __name__ == "__main__":
    main()