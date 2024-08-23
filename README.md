# LombardoTron

I got a fantasy football draft coming up Monday Aug 26, 2024, and I need a way
to rank players.  So I will try out some models in SciKit-Learn that map
year K's stats to year K+1's overall performance.

It's an IDP league, so I gotta pay attention to offense and defense.  Which,
look, this is all the same to me, I'm outsourcing all this to some random
forest anyway.

## Python environment

Here are the actual pip commands I have run in this project's virtual
environment:

```shell
$ pip3 install -U scikit-learn
```

Results:

```shell
$ pip freeze
joblib==1.4.2
numpy==2.1.0
scikit-learn==1.5.1
scipy==1.14.1
threadpoolctl==3.5.0
```

## Data

Much thanks to NFLVerse for these CSVs:

https://github.com/nflverse/nflverse-data/releases/tag/player_stats

I downloaded a few into a folder called `data/` that I told git to ignore.

The column translations are available at:

https://nflreadr.nflverse.com/articles/dictionary_player_stats.html

### Team Changes

How often do players change teams in a regular+post season? Some guys see as
many as five different teams:

```
Mike Evans (00-0031408): 5
Tyreek Hill (00-0033040): 5
JuJu Smith-Schuster (00-0033857): 5
Courtland Sutton (00-0034348): 5
Devin Singletary (00-0035250): 5
Michael Pittman (00-0036252): 5
Ja'Marr Chase (00-0036900): 5
Peyton Hendershot (00-0037569): 5
Tyler Lockett (00-0032211): 4
Demarcus Robinson (00-0032775): 4
George Kittle (00-0033288): 4
Samaje Perine (00-0033526): 4
Austin Ekeler (00-0033699): 4
Evan Engram (00-0033881): 4
River Cracraft (00-0034054): 4
Alec Ingold (00-0035125): 4
Terry McLaurin (00-0035659): 4
Marquise Brown (00-0035662): 4
Brandon Aiyuk (00-0036261): 4
Chase Claypool (00-0036326): 4

Num Teams: Num Players
1: 1359
2: 206
3: 61
4: 17
5: 8
```

About one in six players get traded at least once. So not a totally ignorable
corner case.

### Rookies and Retirement

How many players are in both seasons? Either season? Just '22 or just '23?

```
'22: 1651
'23: 1614
Both: 1189
Either: 2076
Just '22: 462
Just '23: 425
```

So of the 1614 players in '24 I'm gonna rank for the draft, I should expect
440 of them to be rookies, where I can't guess their quality from their previous
NFL stats.