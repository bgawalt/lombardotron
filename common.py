"""Functions used across multiple modules in this suite."""

def empty_float(s: str, default: float = 0.0) -> float:
  """Parse ASCII to a float, and empty strings count as zero."""
  if not s:
    return default
  return float(s)