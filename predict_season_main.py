"""Predict total IDP score earnings of players in 2024 from their 2023 stats."""

import csv
import dataclasses
import hashlib
import sys

import numpy

from sklearn import linear_model # type: ignore

import seasonstats
import weekonestats


NUM_FEATURES = (
  (2 * weekonestats.NUM_WEEK_ONE_FEATURES) + 
  seasonstats.NUM_SEASON_FEATURES
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
  
  @classmethod
  def merge(
    cls,
    first: "LabelledExamples",
    second: "LabelledExamples",
    second_weight_scale: float
  ) -> "LabelledExamples":
    merge_pids = first.pids + second.pids
    merge_feats = numpy.vstack([first.features, second.features])
    merge_labels = first.labels + second.labels
    merge_weights = (
      first.weights +
      tuple(second_weight_scale * wi for wi in second.weights)
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
    vi = numpy.zeros((1, NUM_FEATURES), float)
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
    vi = numpy.zeros((1, NUM_FEATURES), float)
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
    lo = alphas[best_idx - 1] if best_idx != 0 else alphas[best_idx]
    hi = alphas[best_idx + 1] if best_idx != 0 else alphas[best_idx]
  alphas=numpy.logspace(numpy.log10(lo), numpy.log10(hi), num=10)
  rdg = linear_model.RidgeCV(alphas=alphas)
  rdg.fit(train.features, train.labels, train.weights)
  return rdg.alpha_ # type: ignore


def main():
  s21 = seasonstats.SeasonStats(seasonstats.SEASON_FILES_2021, "REG")
  s22 = seasonstats.SeasonStats(seasonstats.SEASON_FILES_2022, "REG")
  s23 = seasonstats.SeasonStats(seasonstats.SEASON_FILES_2023, "REG")
  r21 = weekonestats.WeekOneLeague(weekonestats.ROSTER_FILE_2021)
  r22 = weekonestats.WeekOneLeague(weekonestats.ROSTER_FILE_2022)
  r23 = weekonestats.WeekOneLeague(weekonestats.ROSTER_FILE_2023)
  r24 = weekonestats.WeekOneLeague(weekonestats.ROSTER_FILE_2024)

  s23_from_s22 = build_labelled_examples(
    prev_roster=r22, prev_season=s22, next_roster=r23, next_season=s23)
  s22_from_s21 = build_labelled_examples(
    prev_roster=r21, prev_season=s21, next_roster=r22, next_season=s22)
  s24_from_s23 = build_unlabelled_examples(
    prev_roster=r23, prev_season=s23, next_roster=r24)

  train = LabelledExamples.merge(s23_from_s22, s22_from_s21, 0.9)

  best_alpha = sorted([ridge_param_search(train) for _ in range(21)])[10]
  rdg = linear_model.Ridge(alpha=best_alpha)
  rdg.fit(train.features, train.labels, train.weights)
  preds = rdg.predict(s24_from_s23.features)
  field_names = [
    "pid",
    "full_name",
    "position",
    "team",
    "predicted_idp",
    "drafted",
    "short_name",
  ]
  with open(sys.argv[1], "wt", newline="") as outfile:
    writer = csv.DictWriter(outfile, fieldnames=field_names)
    writer.writeheader()
    for pid, pred in zip(s24_from_s23.pids, preds):
      player = r24.players[pid]
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