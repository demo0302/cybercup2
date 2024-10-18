import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def simulate_data(n_samples=1000):
    np.random.seed(42)
    time = np.arange(n_samples)
    cpu_usage = 50 + 20 * np.sin(2 * np.pi * time / 100) + 5 * np.random.randn(n_samples)
    memory_usage = 70 + 10 * np.sin(2 * np.pi * time / 150) + 3 * np.random.randn(n_samples)
    data = pd.DataFrame({'time': time, 'cpu_usage': cpu_usage, 'memory_usage': memory_usage})
    return data

if __name__ == '__main__':
    data = simulate_data()
    data.plot(x='time', y=['cpu_usage', 'memory_usage'])
    plt.show()
