

def main():
    print('\nSEASON 2023')
    with open('./data/player_stats_season_2022.csv') as infile:
        lines = infile.readlines()
    for i, c in enumerate(lines[0].split(',')):
        print(i, ':', c)
    print('\nDEFENSE')
    with open('./data/player_stats_def_season_2022.csv') as infile:
        lines = infile.readlines()
    for i, c in enumerate(lines[0].split(',')):
        print(i, ':', c)
    raise ValueError('i quit!!')
    print(len(lines), len(lines[0].split(',')))
    pos_to_pts = {}
    for line in lines[1:]:
        spline = line.split(',')
        pos = spline[3]
        pts = float(spline[-2])
        if pos not in pos_to_pts:
            pos_to_pts[pos] = []
        pos_to_pts[pos].append(pts)
    for pos, pts in sorted(pos_to_pts.items(), key=lambda x: len(x[1])):
        print(f'{pos}: {len(pts)}, {sum(pts):0.1f}')


if __name__ == "__main__":
    main()