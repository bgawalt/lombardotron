"""Functions used across multiple modules in this suite."""


# The thirty-two NFL teams.
TEAMS = (
  "ARI", "ATL", "BAL", "BUF", "CAR",  # 0 .. 4
  "CHI", "CIN", "CLE", "DAL", "DEN",  # 5 .. 9
  "DET", "GB", "HOU", "IND", "JAX",  # 10 .. 14
  "KC", "LA", "LAC", "LV", "MIA",  # 15 .. 19
  "MIN", "NE", "NO", "NYG", "NYJ",  # 20 .. 24
  "PHI", "PIT", "SEA", "SF", "TB",  # 25 .. 29
  "TEN", "WAS",  # 30, 31
)

# The twenty-six supported player positions.
POSITIONS = (
  "C", "CB", "DB", "DE", "DT",  # 0 .. 4
  "FB", "FS", "G", "ILB", "K",  # 5 .. 9
  "LB", "MLB", "NT", "OLB", "P",  # 10 .. 14
  "QB", "RB", "SS", "T", "TE",  # 20 .. 24
  "WR",  # 25
)


def empty_float(s: str, default: float = 0.0) -> float:
  """Parse ASCII to a float, and empty strings count as zero."""
  if not s:
    return default
  return float(s)