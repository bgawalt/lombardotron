import csv

import statvalues

_DEF_2022 = "./data/player_stats_def_season_2022.csv"
_DEF_2023 = "./data/player_stats_def_season_2023.csv"
_OFF_2022 = "./data/player_stats_season_2022.csv"
_OFF_2023 = "./data/player_stats_season_2023.csv"


class SeasonStats:

    def __init__(self, off_file: str, def_file: str):

        with open(off_file, 'rt') as infile:
            rows = tuple(r for r in csv.DictReader(infile))
        header_count = {}
        for row in rows:
            for k in row:
                header_count[k] = header_count.get(k, 0) + 1
        for k, v in sorted(header_count.items()):
            print(f'{k}: {v}')


def main():
    for k in statvalues.ALL_FEATURES:
        print(k)


if __name__ == "__main__":
    main()