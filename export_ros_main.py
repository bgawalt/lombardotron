"""Export a CSV for doing Exercise 6.7 of 'Regression and Other Stories.'"""

import csv

import seasonstats


def main():
  s23 = seasonstats.SeasonStats(seasonstats.SEASON_FILES_2023, "REG")
  s24 = seasonstats.SeasonStats(seasonstats.SEASON_FILES_2024, "REG")
  fieldnames = ['player_id', 'player_name', 'idp_2023', 'idp_2024']
  with open('./results/nfl_idp_scores.csv', 'wt') as outfile:
    csv_writer = csv.DictWriter(outfile, fieldnames=fieldnames)
    csv_writer.writeheader()
    for pid in set(s23.player_ids).intersection(s24.player_ids):
      p23 = s23.get_player_stats(pid)
      if p23.idp_score() == 0:
        continue
      p24 = s24.get_player_stats(pid)
      if p24.idp_score() == 0:
        continue
      csv_writer.writerow({
        'player_id': pid,
        'player_name': p23.name,
        'idp_2023': p23.idp_score(),
        'idp_2024': p24.idp_score(),
      })


if __name__ == "__main__":
  main()