import csv

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

class SeasonStats:

    def __init__(self, season: tuple[str, str, str], season_type: str):
        off_file, def_file, kck_file = season
        with open(kck_file, 'rt') as infile:
            for row in csv.DictReader(infile):
                if "kck_games" not in row:
                    raise ValueError("kck_file missing kck_games")
                for k in sorted(row.keys()):
                    print(k)
                break
        player_ids = set([])
        with open(off_file, 'rt') as infile:
            for row in csv.DictReader(infile):
                if "games" not in row:
                    raise ValueError("off_file missing games")
                player_ids.add(row['player_id'])
        with open(def_file, 'rt') as infile:
            for row in csv.DictReader(infile):
                if "def_games" not in row:
                    raise ValueError("def_file missing def_games")
                player_ids.add(row['player_id'])
        self._player_ids = tuple(player_ids)

    @property
    def player_ids(self):
        return self._player_ids


def main():
    print('2022:')
    s22 = SeasonStats(_SEASON_2022, "REG+POST")
    print('\n\n2022:')
    s23 = SeasonStats(_SEASON_2023, "REG")
    p22 = set(s22.player_ids)
    p23 = set(s23.player_ids)
    both = p22.intersection(p23)
    either = p22.union(p23)
    print(f"\n'22: {len(p22)}")
    print(f"'23: {len(p23)}")
    print(f"Both: {len(both)}")
    print(f"Either: {len(either)}")
    print(f"Just '22: {len(p22) - len(both)}")
    print(f"Just '23: {len(p23) - len(both)}")


if __name__ == "__main__":
    main()