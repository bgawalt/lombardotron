import csv
import random

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
_SEASON_TYPE_COLUMN = "season_type"


class SeasonStats:

    def __init__(self, season: tuple[str, str, str], season_type: str):        
        off_file, def_file, kck_file = season
        kck_teams = {}  # {player id : {team : num_games}}
        off_team = {}  # {player_id: team}
        def_teams = {}  # {player id : {team : num_games}}
        names = {} # {player id : name}
        with open(kck_file, "rt") as infile:
            for row in csv.DictReader(infile):
                if row[_SEASON_TYPE_COLUMN] != season_type:
                    continue
                pid = row[_PID_COLUMN]
                name = row[_NAME_COLUMN]
                if pid not in names:
                    names[pid] = name
                elif pid in names and names[pid] != name:
                    raise ValueError(
                        f'Repeat names for {pid}: {name}, {names[pid]}')
                if pid not in kck_teams:
                    kck_teams[pid] = {}
                team = row["team"]
                num_games = int(row["kck_games"])
                kck_teams[pid][team] = kck_teams[pid].get(team, 0) + num_games
        with open(off_file, "rt") as infile:
            for row in csv.DictReader(infile):
                if row[_SEASON_TYPE_COLUMN] != season_type:
                    continue
                pid = row[_PID_COLUMN]
                name = row[_NAME_COLUMN]
                if pid not in names:
                    names[pid] = name
                elif pid in names and names[pid] != name:
                    raise ValueError(
                        f'Repeat names for {pid}: {name}, {names[pid]}')
                if pid in off_team:
                    raise ValueError(f"Wait!! {pid} is in offense twice!!")
                off_team[pid] = row["recent_team"]
        with open(def_file, "rt") as infile:
            for row in csv.DictReader(infile):
                if row[_SEASON_TYPE_COLUMN] != season_type:
                    continue
                pid = row[_PID_COLUMN]
                name = row[_NAME_COLUMN]
                if pid not in def_teams:
                    def_teams[pid] = {}
                if pid not in names:
                    names[pid] = name
                elif pid in names and names[pid] != name:
                    raise ValueError(f'Repeat names for {pid}: {name}, {names[pid]}')
                team = row["team"]
                num_games = int(row["def_games"])
                def_teams[pid][team] = def_teams[pid].get(team, 0) + num_games
        self._player_ids = tuple(
            set(list(kck_teams.keys()) + 
                list(off_team.keys()) + 
                list(def_teams.keys())))
        self._kckteams = kck_teams
        self._offteams = off_team
        self._defteams = def_teams
        self._names = names

    @property
    def player_ids(self):
        return self._player_ids


def main():
    s22 = SeasonStats(_SEASON_2022, "REG+POST")
    all_teams = set([])
    for pid in s22.player_ids:
        for team in s22._defteams.get(pid, {}).keys():
            all_teams.add(team)
        for team in s22._kckteams.get(pid, {}).keys():
            all_teams.add(team)
        all_teams.add(s22._offteams.get(pid, ""))
    for t in sorted(all_teams):
        print(f'\t"{t}",')



if __name__ == "__main__":
    main()