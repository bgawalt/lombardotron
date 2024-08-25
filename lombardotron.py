import collections
import csv

from collections.abc import Iterator

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
_NAME_COLUMN = "player_name"
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
      self._off_games[team] = row["games"]
    elif "def_games" in row:
      if team in self._def_games:
        raise ValueError(f"Multiple insertion, defense, {team}, {self._pid}")
      self._def_games[team] = row["def_games"]
    elif "kck_games" in row:
      if team in self._kck_games:
        raise ValueError(f"Multiple insertion, kicking, {team}, {self._pid}")
      self._kck_games[team] = row["kck_games"]
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


class SeasonStats:
  """Stats for the full league of players, for one season."""

  def __init__(self, season: tuple[str, str, str], season_type: str):
    self._players: dict[str, PlayerStats] = {}
    for filename in season:
      self._add_file(filename=filename, season_type=season_type)

  def _add_file(self, filename: str, season_type: str):
    with open(filename, 'rt') as infile:
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
  
  def get_player_stats(self, player_id: str) -> PlayerStats | None:
    """Stats for the player for this season, if available."""
    return self._players.get(player_id)


def main():
  s22 = SeasonStats(_SEASON_2022, "REG")
  s23 = SeasonStats(_SEASON_2023, "REG")

  for pid in s22.player_ids:
    s22_stats = s22.get_player_stats(pid)
    if s22_stats is None:
      raise ValueError(f"Somehow missing actual {pid} stats from Season '22")
    s23_stats = s23.get_player_stats(pid)
    if s23_stats is None:
      continue
    print(f'{pid}\t{s22_stats.name}\t{s22_stats.idp_score():0.1f}\t' + 
          f'{s23_stats.idp_score():0.1f}')


if __name__ == "__main__":
    main()