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

## Building Examples

Machine learning requires labeled examples; `(x; y)` pairs, for feature vectors
`x` and labels `y` (scalar or vector).

### Example unit

Each example is a player who played in both the 2022-23 season *and* the
2023-24 season. That comes to 1,218 players.

### Labels

This is straightforward: the label for a player will be their IDP score,
calculated over the player's total stats accumulated for the full '23 *regular*
season (no post season).

### Feature vectors

Feature vectors are based on the player's stats from '22, accumulated, again,
across the *regular* season games.

#### Numerical stats

The features I will use for each player are their season-wide stats racked up in
the 2022-23 regular season.

The list of numerical stats I extract from the NFLverse CSVs are in `statvalues.py`.
I wind up with 97 of them overall.  Interestingly, one stats counts a player's
field goals missed from 0 to 19 yards away from the end zone -- and not a single
player ever missed an FG from that close in the '22-'23 season. So that feature
is useless.

[Commit with this code: 1104d60](https://github.com/bgawalt/lombardotron/blob/1104d6095c098b4932e42be877890b868cd80b2b/lombardotron.py)

#### Text stats

TODO: Team, position

### Example weight

I can give each example an individual weight, to tell whatever predictive
algorithm instructions on how much importance it should put on correctly
predicting *that* example, vs. its peers.

Each team in my league has 19 slots, and there's 12 league members, so that's
228 drafted players. If you look at the 250th highest IDP score in 2023, that's
around 117 points. So most of the players I'm interested in have an IDP score
above 100 points.

My example-weighting function weights each player based on that threshold of 100
IDP points:

1.  Everyone with a '23 IDP score of 100 or more gets a weight of 1
2.  Everyone else gets a weight of their IDP divided by 100

So as a player's score gets farther and farther from the threshold of "likely to
be on a team in my fantasy leagure," its influence on model training steadily
diminishes.

This way, the rules I learn, will be focused on correctly predicting the top
quantile of the league. It doesn't do me any good to build a model that does
great predicting perfomrance players that only produce a small number of points:
they're not the ones I'm gonna need on my team if I'm gonna win.

This starts showing up in the modeling in Round 3.

[Commit with this code: ec8a2ae](https://github.com/bgawalt/lombardotron/blob/ec8a2aeee3728c8840acaac0ed30f9d19fa98a82/lombardotron.py)

## Modeling

### Round 1: (Overfit) Ordinary Least Squares

We can try out the first attempt at a learned model for mapping season K's stats
to season K+1: by fitting ordinary least squares to the 97 '22 season
*numerical* stats (I'd not yet looped in team or position indicators) for
the 1218 players on record, so that our guess at '23 IDP is a weighted sum of
those stats.

Note: I trained this model *before* arriving at my example weight scheme; this
predictor was trained treating every example-player as equally important to
correctly guess.

We can overlay the OLS predictions on top of the previous scatter plot, where
'22 IDP was used as the horizontal-axis predictor:

![Two predictors of '23 IDP: Overfit OLS and '22 IDP](fig/ols_v_s22idp.png)

The new, OLS result, in red, looks slightly tighter. (I haven't started checking
actual squared error loss metrics yet.)

But! This is cheating. *The weights have been specifically picked to fit the
2023-24 IDP score outcomes.** These are not predictions on held-out data, the
way the weights that comprise IDP were set before the '23 season outcomes were
known.

This estimator is also unregularized, which I think explains wackiness around
why both "field goals made" and "field goals missed" have the same positive
weight parameter. Missing a field goal should not be as good as making one!

[Commit with this code: 24b17b2](https://github.com/bgawalt/lombardotron/blob/24b17b2bd80342f5a9e302dbb92eef32ce738ecb/lombardotron.py)

### Round 2: (Also mostly overfit) Support Vector Regression

Note: I do now have "weight by clipping `0.01 * idp` to a max of 1.0" sample
weighting in place.

I can repeat the above exercise with a more complicated, nonlinear model:
support vector regression.  I do set the regularization parameter of the model
with a 7-fold cross-validation routine, so there's *some* constraint on how
low the training error can go before totally detonating generalizability.

But this scatter plot still represents fitting that SVR with the cross-validated
regression parameter to the full league and *then* asking it for predictions of
the player's '23 point totals, *after* already being shown them at training
time.

![Two predictors of '23 IDP: Overfit SVR and '22 IDP](fig/svr_v_s22idp.png)

In other words, (a) this is still a chart of training set error, who knows how
much worse test-set error would be; and (b) this error still seems very high!!
The prediction just seems fundamentally difficult with this feature set.

(I do still need to add text features into the feature set, i.e., note each
player's position.)

[Commit with this code: 316d138](https://github.com/bgawalt/lombardotron/blob/316d13833bc2f57d9a87755bd57735afdcf4c98d/lombardotron.py)