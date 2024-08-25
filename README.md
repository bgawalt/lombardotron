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

[Commit with this code: 3cfbe72](https://github.com/bgawalt/lombardotron/blob/3cfbe7256c98c3e22598538eb114b79a2862df5f/lombardotron.py)

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

[Commit with this code: af9dd8c](https://github.com/bgawalt/lombardotron/blob/af9dd8c38c797e504103d775180cc4613190632d/lombardotron.py)

### Fantasy Points

Here's the 10 lowest- and 10 highest-scoring players in the 2022-23 season,
according to my IDP league's scoring system:

```
T.Boyle (00-0034177):   -2.9
M.Dickson (00-0034160): -1.8
C.Beathard (00-0033936):        -1.0
C.Henne (00-0026197):   -0.5
N.Sudfeld (00-0032792): -0.4
J.Gillan (00-0035042):  -0.3
J.Gordon (00-0029664):  0.0
B.Anger (00-0029692):   0.0
D.Bakhtiari (00-0030074):       0.0
T.Gentry (00-0033633):  0.0
...
J.Jacobs (00-0035700):  331.3
D.Adams (00-0031381):   337.0
T.Hill (00-0033040):    347.2
J.Burrow (00-0036442):  350.7
C.McCaffrey (00-0033280):       356.4
J.Jefferson (00-0036322):       371.7
J.Hurts (00-0036389):   381.0
A.Ekeler (00-0033699):  382.7
J.Allen (00-0034857):   403.3
P.Mahomes (00-0033873): 418.9
```

[Commit with this code: 4f85564](https://github.com/bgawalt/lombardotron/blob/4f85564ee9b8755b31f015436ca9e41e2fb3f5d6/lombardotron.py)

We can also look at each player's total IDP points across '22 vs. across '23
(with a series of y = x equality included in red):

![IDP Scatter Plot, 2023-24 v 2022-23 Seasons](fig/idp_scatter.png)

**This is the baseline to beat.** I will need to come up with a function of
'22-season stats that is a tighter match to '23 performance, than just applying
the IDP formula to the '22 stats and calling that my prediction.

Here are histograms of season-long player IDP score, with y-axis counts in
linear and log scale, for the 2023-24 season:

![IDP Score Histogram](fig/idp_histogram.png)

![IDP Score Histogram, Log Scale](fig/idp_histogram_log.png)

[Commit with this code: 1600d74](https://github.com/bgawalt/lombardotron/blob/1600d74f4f316309844f654d4dd0a97ff325bfba/lombardotron.py)