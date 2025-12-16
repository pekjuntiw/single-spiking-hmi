import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


ups = [f'up_{i}.csv' for i in range(1, 7)]
downs = [f'down_{i}.csv' for i in range(1, 7)]
lefts = [f'left_{i}.csv' for i in range(1, 7)]
rights = [f'right_{i}.csv' for i in range(1, 7)]

ups_start = [400, 280, 300, 300, 250, 400]
downs_start = [290, 100, 200, 150, 200, 200]
lefts_start = [480, 250, 290, 300, 350, 300]
rights_start = [250, 300, 150, 200, 250, 75]

ups_end = [3800, 3800, 4100, 4000, 3600, 4650]
downs_end = [3700, 3700, 3900, 3800, 4000, 3800]
lefts_end = [4300, 3900, 4000, 4050, 4200, 4000]
rights_end = [4200, 4200, 4100, 4200, 4000, 4100]

print([i - j for i, j in zip(ups_end, ups_start)])
print([i - j for i, j in zip(downs_end, downs_start)])
print([i - j for i, j in zip(lefts_end, lefts_start)])
print([i - j for i, j in zip(rights_end, rights_start)])

ups_combined = None
downs_combined = None
lefts_combined = None
rights_combined = None

for f, s, e in zip(ups, ups_start, ups_end):
    data = pd.read_csv(f).to_numpy()[s:e]
    if ups_combined is None:
        ups_combined = data
    else:
        ups_combined = np.concatenate([ups_combined, data])
for f, s, e in zip(downs, downs_start, downs_end):
    data = pd.read_csv(f).to_numpy()[s:e]
    if downs_combined is None:
        downs_combined = data
    else:
        downs_combined = np.concatenate([downs_combined, data])
for f, s, e in zip(lefts, lefts_start, lefts_end):
    data = pd.read_csv(f).to_numpy()[s:e]
    if lefts_combined is None:
        lefts_combined = data
    else:
        lefts_combined = np.concatenate([lefts_combined, data])
for f, s, e in zip(rights, rights_start, rights_end):
    data = pd.read_csv(f).to_numpy()[s:e]
    if rights_combined is None:
        rights_combined = data
    else:
        rights_combined = np.concatenate([rights_combined, data])

rng = np.random.default_rng(seed=20240606)
up_index = np.sort(rng.choice(np.arange(len(ups_combined)), 500, replace=False))
down_index = np.sort(rng.choice(np.arange(len(downs_combined)), 500, replace=False))
left_index = np.sort(rng.choice(np.arange(len(lefts_combined)), 500, replace=False))
right_index = np.sort(rng.choice(np.arange(len(rights_combined)), 500, replace=False))

up_sampled = ups_combined[up_index]
down_sampled = downs_combined[down_index]
left_sampled = lefts_combined[left_index]
right_sampled = rights_combined[right_index]

pd.DataFrame(
    up_sampled,
    columns=['timestamp', 'Channel 1', 'Channel 2', 'Channel 3', 'Channel 4', 'Channel 5', 'Channel 6']
).to_csv('generated/up.csv', index=False)
pd.DataFrame(
    down_sampled,
    columns=['timestamp', 'Channel 1', 'Channel 2', 'Channel 3', 'Channel 4', 'Channel 5', 'Channel 6']
).to_csv('generated/down.csv', index=False)
pd.DataFrame(
    left_sampled,
    columns=['timestamp', 'Channel 1', 'Channel 2', 'Channel 3', 'Channel 4', 'Channel 5', 'Channel 6']
).to_csv('generated/left.csv', index=False)
pd.DataFrame(
    right_sampled,
    columns=['timestamp', 'Channel 1', 'Channel 2', 'Channel 3', 'Channel 4', 'Channel 5', 'Channel 6']
).to_csv('generated/right.csv', index=False)

plt.figure()
plt.plot(np.arange(len(ups_combined)), ups_combined[:, 2])
plt.plot(np.arange(len(ups_combined)), ups_combined[:, 2])
plt.plot(np.arange(len(ups_combined)), ups_combined[:, 3])
plt.plot(np.arange(len(ups_combined)), ups_combined[:, 4])
plt.plot(np.arange(len(ups_combined)), ups_combined[:, 5])
plt.plot(np.arange(len(ups_combined)), ups_combined[:, 6])
plt.ylim([-50, 750])
plt.title("UP")

plt.figure()
plt.plot(np.arange(len(downs_combined)), downs_combined[:, 2])
plt.plot(np.arange(len(downs_combined)), downs_combined[:, 2])
plt.plot(np.arange(len(downs_combined)), downs_combined[:, 3])
plt.plot(np.arange(len(downs_combined)), downs_combined[:, 4])
plt.plot(np.arange(len(downs_combined)), downs_combined[:, 5])
plt.plot(np.arange(len(downs_combined)), downs_combined[:, 6])
plt.ylim([-50, 750])
plt.title("DOWN")

plt.figure()
plt.plot(np.arange(len(lefts_combined)), lefts_combined[:, 2])
plt.plot(np.arange(len(lefts_combined)), lefts_combined[:, 2])
plt.plot(np.arange(len(lefts_combined)), lefts_combined[:, 3])
plt.plot(np.arange(len(lefts_combined)), lefts_combined[:, 4])
plt.plot(np.arange(len(lefts_combined)), lefts_combined[:, 5])
plt.plot(np.arange(len(lefts_combined)), lefts_combined[:, 6])
plt.ylim([-50, 750])
plt.title("LEFT")

plt.figure()
plt.plot(np.arange(len(rights_combined)), rights_combined[:, 2])
plt.plot(np.arange(len(rights_combined)), rights_combined[:, 2])
plt.plot(np.arange(len(rights_combined)), rights_combined[:, 3])
plt.plot(np.arange(len(rights_combined)), rights_combined[:, 4])
plt.plot(np.arange(len(rights_combined)), rights_combined[:, 5])
plt.plot(np.arange(len(rights_combined)), rights_combined[:, 6])
plt.title("RIGHT")
plt.ylim([-50, 750])

plt.figure()
plt.plot(np.arange(len(up_sampled)), up_sampled[:, 2])
plt.plot(np.arange(len(up_sampled)), up_sampled[:, 2])
plt.plot(np.arange(len(up_sampled)), up_sampled[:, 3])
plt.plot(np.arange(len(up_sampled)), up_sampled[:, 4])
plt.plot(np.arange(len(up_sampled)), up_sampled[:, 5])
plt.plot(np.arange(len(up_sampled)), up_sampled[:, 6])
plt.ylim([-50, 750])
plt.title("UP")

plt.figure()
plt.plot(np.arange(len(down_sampled)), down_sampled[:, 2])
plt.plot(np.arange(len(down_sampled)), down_sampled[:, 2])
plt.plot(np.arange(len(down_sampled)), down_sampled[:, 3])
plt.plot(np.arange(len(down_sampled)), down_sampled[:, 4])
plt.plot(np.arange(len(down_sampled)), down_sampled[:, 5])
plt.plot(np.arange(len(down_sampled)), down_sampled[:, 6])
plt.ylim([-50, 750])
plt.title("DOWN")

plt.figure()
plt.plot(np.arange(len(left_sampled)), left_sampled[:, 2])
plt.plot(np.arange(len(left_sampled)), left_sampled[:, 2])
plt.plot(np.arange(len(left_sampled)), left_sampled[:, 3])
plt.plot(np.arange(len(left_sampled)), left_sampled[:, 4])
plt.plot(np.arange(len(left_sampled)), left_sampled[:, 5])
plt.plot(np.arange(len(left_sampled)), left_sampled[:, 6])
plt.ylim([-50, 750])
plt.title("LEFT")

plt.figure()
plt.plot(np.arange(len(right_sampled)), right_sampled[:, 2])
plt.plot(np.arange(len(right_sampled)), right_sampled[:, 2])
plt.plot(np.arange(len(right_sampled)), right_sampled[:, 3])
plt.plot(np.arange(len(right_sampled)), right_sampled[:, 4])
plt.plot(np.arange(len(right_sampled)), right_sampled[:, 5])
plt.plot(np.arange(len(right_sampled)), right_sampled[:, 6])
plt.title("RIGHT")
plt.ylim([-50, 750])

plt.show()


'''up = pd.read_csv("up_long.csv").drop(columns=['Channel 6'])
down = pd.read_csv("down_long.csv").drop(columns=['Channel 6'])
left = pd.read_csv("left_long.csv").drop(columns=['Channel 6'])
right = pd.read_csv("right_long.csv").drop(columns=['Channel 6'])

print(len(up), len(down), len(left), len(right))

rng = np.random.default_rng(seed=20240126)
up_index = np.sort(rng.choice(np.arange(375, 5400), 500, replace=False))
down_index = np.sort(rng.choice(np.arange(500, 5400), 500, replace=False))
left_index = np.sort(rng.choice(np.arange(200, 5400), 500, replace=False))
right_index = np.sort(rng.choice(np.arange(300, 5200), 500, replace=False))

up.iloc[up_index].to_csv('generated/up.csv', index=False)
down.iloc[down_index].to_csv('generated/down.csv', index=False)
left.iloc[left_index].to_csv('generated/left.csv', index=False)
right.iloc[right_index].to_csv('generated/right.csv', index=False)

plt.figure()
plt.title("Wrist UP")
plt.plot(np.arange(len(up)), up['Channel 1'])
plt.plot(np.arange(len(up)), up['Channel 2'])
plt.plot(np.arange(len(up)), up['Channel 3'])
plt.plot(np.arange(len(up)), up['Channel 4'])
plt.plot(np.arange(len(up)), up['Channel 5'])
plt.vlines(up_index, 0, -200, linestyles='dashed')

plt.figure()
plt.title("Wrist DOWN")
plt.plot(np.arange(len(down)), down['Channel 1'])
plt.plot(np.arange(len(down)), down['Channel 2'])
plt.plot(np.arange(len(down)), down['Channel 3'])
plt.plot(np.arange(len(down)), down['Channel 4'])
plt.plot(np.arange(len(down)), down['Channel 5'])
plt.vlines(down_index, 0, -200, linestyles='dashed')

plt.figure()
plt.title("Wrist LEFT")
plt.plot(np.arange(len(left)), left['Channel 1'])
plt.plot(np.arange(len(left)), left['Channel 2'])
plt.plot(np.arange(len(left)), left['Channel 3'])
plt.plot(np.arange(len(left)), left['Channel 4'])
plt.plot(np.arange(len(left)), left['Channel 5'])
plt.vlines(left_index, 0, -200, linestyles='dashed')

plt.figure()
plt.title("Wrist RIGHT")
plt.plot(np.arange(len(right)), right['Channel 1'])
plt.plot(np.arange(len(right)), right['Channel 2'])
plt.plot(np.arange(len(right)), right['Channel 3'])
plt.plot(np.arange(len(right)), right['Channel 4'])
plt.plot(np.arange(len(right)), right['Channel 5'])
plt.vlines(right_index, 0, -200, linestyles='dashed')

plt.figure()
plt.title("Wrist UP, randomly selected timestamps")
plt.plot(np.arange(len(up.iloc[up_index])), up.iloc[up_index]['Channel 1'])
plt.plot(np.arange(len(up.iloc[up_index])), up.iloc[up_index]['Channel 2'])
plt.plot(np.arange(len(up.iloc[up_index])), up.iloc[up_index]['Channel 3'])
plt.plot(np.arange(len(up.iloc[up_index])), up.iloc[up_index]['Channel 4'])
plt.plot(np.arange(len(up.iloc[up_index])), up.iloc[up_index]['Channel 5'])

plt.figure()
plt.title("Wrist DOWN, randomly selected timestamps")
plt.plot(np.arange(len(down.iloc[down_index])), down.iloc[down_index]['Channel 1'])
plt.plot(np.arange(len(down.iloc[down_index])), down.iloc[down_index]['Channel 2'])
plt.plot(np.arange(len(down.iloc[down_index])), down.iloc[down_index]['Channel 3'])
plt.plot(np.arange(len(down.iloc[down_index])), down.iloc[down_index]['Channel 4'])
plt.plot(np.arange(len(down.iloc[down_index])), down.iloc[down_index]['Channel 5'])

plt.figure()
plt.title("Wrist LEFT, randomly selected timestamps")
plt.plot(np.arange(len(left.iloc[left_index])), left.iloc[left_index]['Channel 1'])
plt.plot(np.arange(len(left.iloc[left_index])), left.iloc[left_index]['Channel 2'])
plt.plot(np.arange(len(left.iloc[left_index])), left.iloc[left_index]['Channel 3'])
plt.plot(np.arange(len(left.iloc[left_index])), left.iloc[left_index]['Channel 4'])
plt.plot(np.arange(len(left.iloc[left_index])), left.iloc[left_index]['Channel 5'])

plt.figure()
plt.title("Wrist RIGHT, randomly selected timestamps")
plt.plot(np.arange(len(right.iloc[right_index])), right.iloc[right_index]['Channel 1'])
plt.plot(np.arange(len(right.iloc[right_index])), right.iloc[right_index]['Channel 2'])
plt.plot(np.arange(len(right.iloc[right_index])), right.iloc[right_index]['Channel 3'])
plt.plot(np.arange(len(right.iloc[right_index])), right.iloc[right_index]['Channel 4'])
plt.plot(np.arange(len(right.iloc[right_index])), right.iloc[right_index]['Channel 5'])

plt.show()

'''