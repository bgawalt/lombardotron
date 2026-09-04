import csv

import seasonstats

DATA_2025 = "./data/stats_player_reg_2025.csv"


def main():
  rows = []
  with open(DATA_2025, 'rt') as infile:
    reader = csv.DictReader(infile)
    for row in reader:
      rows.append(row)
      break

  for k in row:
    print(k)
  return

  for k in row:
    if k not in seasonstats.SEASON_STAT_FEATURES:
      print(f'csv had {k} but PREDICTORS didnt')
  for k in seasonstats.SEASON_STAT_FEATURES:
    if k not in row:
      print(f'PREDICTORS had {k} but csv didnt')


if __name__ == '__main__':
  main()