import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


ups = [f'up_{i}.csv' for i in range(1, 7)]
downs = [f'down_{i}.csv' for i in range(1, 7)]
lefts = [f'left_{i}.csv' for i in range(1, 7)]
rights = [f'right_{i}.csv' for i in range(1, 7)]

for f in (ups + downs + lefts + rights):
    data = pd.read_csv(f)
    plt.figure()
    plt.title(f)
    plt.plot(np.arange(len(data)), data['Channel 1'], label='CH1')
    plt.plot(np.arange(len(data)), data['Channel 2'], label='CH2')
    plt.plot(np.arange(len(data)), data['Channel 3'], label='CH3')
    plt.plot(np.arange(len(data)), data['Channel 4'], label='CH4')
    plt.plot(np.arange(len(data)), data['Channel 5'], label='CH5')
    plt.plot(np.arange(len(data)), data['Channel 6'], label='CH6')
    plt.legend()
plt.show()
