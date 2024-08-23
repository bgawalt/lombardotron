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