import csv

import statvalues

_DEF_2022 = "./data/player_stats_def_season_2022.csv"
_DEF_2023 = "./data/player_stats_def_season_2023.csv"
_OFF_2022 = "./data/player_stats_season_2022.csv"
_OFF_2023 = "./data/player_stats_season_2023.csv"


# TODO: move REG, POST, REG+POST to statvalues enum

class SeasonStats:

    def __init__(self, off_file: str, def_file: str, season_type_filter: str):
        player_id_counts = {}
        player_id_name = {}
        with open(off_file, 'rt') as infile:
            for row in csv.DictReader(infile):
                if row['season_type'] != season_type_filter:
                    continue
                player = row['player_id']
                name = row['player_display_name']
                if player in player_id_name:
                    if player_id_name[player] != name:
                        raise ValueError(
                            f'{player}, {name}, {player_id_name[player]}')
                else:
                    player_id_name[player] = name
                player_id_counts[player] = player_id_counts.get(player, 0) + 1
        with open(def_file, 'rt') as infile:
            for row in csv.DictReader(infile):                    
                if row['season_type'] != season_type_filter:
                    continue
                player = row['player_id']
                player_id_counts[player] = player_id_counts.get(player, 0) + 1
                name = row['player_display_name']
                if player in player_id_name:
                    if player_id_name[player] != name:
                        raise ValueError(
                            f'{player}, {name}, {player_id_name[player]}')
                else:
                    player_id_name[player] = name
        for p, c in sorted(player_id_counts.items(),
                           key=lambda x: x[1], reverse=True)[:20]:
            n = player_id_name[p]
            print(f'{n} ({p}): {c}')
        count_counts = {}
        for c in player_id_counts.values():
            count_counts[c] = count_counts.get(c, 0) + 1
        print('\n')
        for c in sorted(count_counts.keys()):
            print(f'{c}: {count_counts[c]}')


def main():
    SeasonStats(_OFF_2022, _DEF_2022, "REG+POST")


if __name__ == "__main__":
    main()