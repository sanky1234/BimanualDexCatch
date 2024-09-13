import os, glob
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.python.framework import tensor_util

import pandas as pd
import matplotlib.pyplot as plt
import yaml

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


def draw_learning_curve(target_list):
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


def plot_combined_boxplot(data, target_keys):
    plt.figure(figsize=(12, 8))

    # Make sure to return a dictionary so we can access the elements, and disable outliers (fliers)
    boxplot = data.boxplot(patch_artist=True, return_type='dict', showfliers=False)

    # Set colors for the boxes
    colors = ['lightblue', 'lightgreen', 'lightcoral', 'lightpink', 'lightyellow']

    # Apply colors to the boxes and adjust line thickness
    for patch, color in zip(boxplot['boxes'], colors * (len(boxplot['boxes']) // len(colors) + 1)):
        patch.set_facecolor(color)
        patch.set_linewidth(2)  # Thicker lines for boxes

    # Thicken the lines of whiskers and caps
    for whisker in boxplot['whiskers']:
        whisker.set_linewidth(2)
    for cap in boxplot['caps']:
        cap.set_linewidth(2)

    # Thicken the lines for medians and set them to a distinctive color
    for median in boxplot['medians']:
        median.set_linewidth(2.5)
        median.set_color('blue')

    # Calculate and mark averages
    means = data.mean()  # Calculate the mean for each column
    for i, mean in enumerate(means):
        plt.text(i + 1, mean, f'{mean:.2f}', horizontalalignment='center', color='black', weight='bold', fontsize=12)

    # Set custom X-axis labels with target_keys and increase font size
    plt.xticks(ticks=range(1, len(target_keys) + 1), labels=target_keys, rotation=45, fontsize=12)

    # Add gridlines and set their style
    plt.grid(True, linestyle='--', alpha=0.7)

    plt.title('Catcher Reward at Noise Scale 1.0', fontsize=16)

    plt.tight_layout()
    plt.show()

def draw_eval_plot(target_dict):
    print("target_dict: ", target_dict)

    current_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.join(current_dir, '../isaacgymenvs/runs')
    model_dirs = os.listdir(root_dir)

    only1 = True

    combined_data = pd.DataFrame()  # Empty DataFrame to hold combined data

    target_keys = []
    for model_dir in model_dirs:
        if model_dir in target_dict.values():
            target_key = next(key for key, value in target_dict.items() if value == model_dir)
            target_dir = os.path.join(root_dir, model_dir, 'eval')
            _csvs = os.listdir(target_dir)

            # sort as [ns1.0, ns1.2, ns1.5]
            csvs = sorted(_csvs, key=lambda x: float(x.split('_ns')[1].split('_')[0]))
            print("key: {}, file: {} ".format(target_key, csvs))
            target_keys.append(target_key)

            for csv in csvs:
                csv_path = os.path.join(target_dir, csv)
                data = pd.read_csv(csv_path)
                print(f"Reading CSV: {csv_path}")

                # Combine the current CSV with the overall data
                combined_data = pd.concat([combined_data, data], axis=1)
                if only1:
                    break

    # Plot combined boxplot if there is data
    if not combined_data.empty:
        custom_order = ['SA', 'SA_pbt', 'MA_fix_0.9', 'MA_decay_0.5', 'HA_fix_0.9']
        print("custom_order: ", custom_order)
        print("target_keys: ", target_keys)
        target_keys_sorted = sorted(target_keys, key=lambda x: custom_order.index(x))
        print("target_keys_sorted: ", target_keys_sorted)
        index_map = [target_keys.index(key) for key in custom_order]  # custom_order에 맞는 인덱스 찾기

        # 열 값을 스왑하여 custom_order 순서에 맞게 재배치
        combined_data.iloc[:, :] = combined_data.iloc[:, index_map].values

        # combined_data.iloc[:, [0, 3]] = combined_data.iloc[:, [3, 0]].values
        plot_combined_boxplot(combined_data, target_keys_sorted)


if __name__ == '__main__':
    # target_list = ['MA_BimanualDexCatchUR3Allegro_2024-09-10_09-23-44',
    #                'MA_BimanualDexCatchUR3Allegro_2024-09-10_18-13-24']

    # draw_learning_curve(target_list=target_list)

    current_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.join(current_dir, '../isaacgymenvs/evaluation')
    map_path = os.path.join(root_dir, 'experimentMapAll.yaml')

    with open(map_path, 'r') as file:
        target_map_dict = yaml.safe_load(file)['map']

    target_tags = ['SA', 'SA_pbt', 'MA_fix_0.9', 'MA_decay_0.5', 'HA_fix_0.9']
    target_dict = {key: target_map_dict[key] for key in target_tags if key in target_map_dict}
    draw_eval_plot(target_dict=target_dict)
