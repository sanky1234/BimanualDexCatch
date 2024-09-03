import os, glob
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.python.framework import tensor_util

class SmoothedPlot:
    def __init__(self, smoothing_factor=0.6):
        self.smoothing_factor = smoothing_factor
        self.data_list = []
        self.labels = []

    def smooth_values(self, values):
        smoothed = []
        for i in range(len(values)):
            if i == 0:
                smoothed.append(values[i])
            else:
                smoothed_value = smoothed[-1] * (1 - self.smoothing_factor) + values[i] * self.smoothing_factor
                smoothed.append(smoothed_value)
        return smoothed

    def load_data(self, iterations, rewards, label=None):
        smoothed_values = self.smooth_values(rewards)
        self.data_list.append((iterations, rewards, smoothed_values))
        self.labels.append(label)

    def plot(self):
        plt.figure(figsize=(10, 6))

        for (step, raw_values, smoothed_values), label in zip(self.data_list, self.labels):
            # Raw data (faint)
            plt.plot(step, raw_values, alpha=0.3, label=f'Raw {label}')
            # Smoothed data (vivid)
            plt.plot(step, smoothed_values, label=f'Smoothed {label}')

        plt.title('Bimanual Throw/Catch Reward Plot')
        plt.xlabel('Epoch')
        plt.ylabel('Reward')
        plt.grid(True)
        plt.legend()
        plt.show()

if __name__ == '__main__':
    target_list = ['MA_BimanualDexCatchUR3Allegro_2024-08-29_17-56-35',
                   'SA_BimanualDexCatchUR3Allegro_2024-08-30_12-11-09']

    current_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.join(current_dir, '../isaacgymenvs/runs')
    model_dirs = os.listdir(root_dir)

    plotter = SmoothedPlot(smoothing_factor=0.6)

    for model_dir in model_dirs:
        if model_dir in target_list:
            _path = os.path.join(root_dir, model_dir, 'summaries')
            summary = os.listdir(_path)[0]
            summary_path = os.path.join(_path, summary)

            iterations = []
            rewards = []
            for event in tf.compat.v1.train.summary_iterator(summary_path):
                for value in event.summary.value:
                    if value.tag == 'rewards/iter':
                        if value.HasField('simple_value'):
                            iterations.append(event.step)
                            rewards.append(value.simple_value)

            label = model_dir.split('_')[0]
            plotter.load_data(iterations, rewards, label)

    plotter.plot()
