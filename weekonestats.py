"""Describe players based on how they appear on Week 1 of a season."""

import csv
import dataclasses
import datetime

import numpy

import common


ROSTER_FILE_2021 = "./data/roster_weekly_2022.csv"
ROSTER_FILE_2022 = "./data/roster_weekly_2022.csv"
ROSTER_FILE_2023 = "./data/roster_weekly_2023.csv"
ROSTER_FILE_2024 = "./data/roster_weekly_2024.csv"

NUM_WEEK_ONE_FEATURES = 8

WEEK_ONE_FEATURES = (
  "active",
  "age",
  "height",
  "weight",
  "years_exp",
  "years_since_entry",
  "years_since_rookie",
  "draft_number"
)


@dataclasses.dataclass(frozen=True)
class WeekOnePlayer:
  """Details about a player just before a season's Week 1 kickoff."""
  pid: str
  name: str
  short_name: str
  team: str

  position: str
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
    float).reshape((1, NUM_WEEK_ONE_FEATURES))

  @classmethod
  def from_row(cls, row: dict[str, str]) -> 'WeekOnePlayer | None':
    now = datetime.datetime.now()
    pid = row["gsis_id"]
    if not pid:
      return None
    if int(row["week"]) != 1:
      return None
    if not row["birth_date"]:
      return None
    birth_date = datetime.datetime.strptime(row["birth_date"], "%Y-%m-%d")
    age = (now - birth_date).days / 365
    entry_date = datetime.datetime(year=int(row["entry_year"]), month=9, day=1)
    entry_age = (now - entry_date).days / 365
    rook_date = datetime.datetime(year=int(row["rookie_year"]), month=9, day=1)
    rook_age = (now - rook_date).days / 365
    "entry_year,rookie_year,draft_club,draft_number"
    return WeekOnePlayer(
      pid=pid,
      name=row["full_name"],
      short_name=f'{row["first_name"][0]}.{row["last_name"]}',
      team=row["team"],
      position=row["position"],
      active=(row["status"] == "ACT"),
      age=age,
      height=float(row["height"]),
      weight=float(row["weight"]),
      years_exp=float(row["years_exp"]),
      entry_age=entry_age,
      rookie_age=rook_age,
      draft_number=common.empty_float(row["draft_number"], default=400)
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