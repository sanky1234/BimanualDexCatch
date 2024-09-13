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


import matplotlib.pyplot as plt

# Function to plot combined boxplot with enhanced Y-axis label styling
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
        plt.text(i + 1, mean, f'{mean:.2f}', horizontalalignment='center', color='black', weight='bold', fontsize=20)

    # Set custom X-axis labels with target_keys and increase font size
    plt.xticks(ticks=range(1, len(target_keys) + 1), labels=target_keys, rotation=45, fontsize=24)

    # Y-axis settings: increase font size, rotate label, and add 'Reward'
    plt.yticks(fontsize=16)  # Increase font size for Y-axis numbers
    plt.ylabel('Reward', fontsize=20, rotation=90)  # Y-axis label

    # Add gridlines and set their style
    plt.grid(True, linestyle='--', alpha=0.7)

    # Set plot title
    plt.title('Catcher Reward at Noise Scale 1.0', fontsize=24)

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
        combined_data.iloc[:, :] = combined_data.iloc[:, index_map].values
        plot_combined_boxplot(combined_data, target_keys_sorted)


# Adjust pandas display options to avoid truncation
pd.set_option('display.max_colwidth', None)  # Prevent truncating long column names
pd.set_option('display.max_rows', None)      # Show all rows without truncating
pd.set_option('display.max_columns', None)   # Show all columns without truncating


def print_mean_std(target_dict, custom_order):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.join(current_dir, '../isaacgymenvs/runs')
    model_dirs = os.listdir(root_dir)

    results = []  # Store results as a list of dictionaries

    for model_dir in model_dirs:
        if model_dir in target_dict.values():
            target_key = next(key for key, value in target_dict.items() if value == model_dir)
            target_dir = os.path.join(root_dir, model_dir, 'eval')
            _csvs = os.listdir(target_dir)

            # sort as [ns1.0, ns1.2, ns1.5]
            csvs = sorted(_csvs, key=lambda x: float(x.split('_ns')[1].split('_')[0]))
            print(f"Processing key: {target_key}")

            # Read and process each CSV file
            for csv_file in csvs:
                csv_path = os.path.join(target_dir, csv_file)
                data = pd.read_csv(csv_path)  # Load the CSV data into a DataFrame

                # Calculate mean and std for the entire DataFrame
                mean_value = data.values.mean()  # Overall mean across all columns
                std_value = data.values.std()  # Overall standard deviation across all columns

                # Append the results as a dictionary, attaching the key to the CSV file name
                results.append({
                    'CSV File': f"{csv_file} (Key: {target_key})",  # Add key to CSV file name
                    'Key': target_key,  # For sorting purposes
                    'Mean': round(mean_value, 2),
                    'Standard Deviation': round(std_value, 2)
                })

    # Convert the results list into a DataFrame for tabular display
    result_df = pd.DataFrame(results)

    # Reorder based on custom_order
    result_df['Key'] = pd.Categorical(result_df['Key'], categories=custom_order, ordered=True)
    result_df = result_df.sort_values('Key')  # Sort based on custom_order

    # Drop the 'Key' column from the final output
    result_df = result_df.drop(columns=['Key'])

    # Format the DataFrame for single-line output for each row
    pd.set_option('display.expand_frame_repr', False)  # Prevent splitting into multiple lines

    # Print the DataFrame as a table
    print("\nResults in tabular format (sorted by custom order):")
    print(result_df)


if __name__ == '__main__':
    # target_list = ['MA_BimanualDexCatchUR3Allegro_2024-09-10_09-23-44',
    #                'MA_BimanualDexCatchUR3Allegro_2024-09-10_18-13-24']

    # draw_learning_curve(target_list=target_list)

    current_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.join(current_dir, '../isaacgymenvs/evaluation')
    map_path = os.path.join(root_dir, 'experimentMapAll.yaml')

    with open(map_path, 'r') as file:
        target_map_dict = yaml.safe_load(file)['map']

    target_tags = ['SA', 'SA_pbt',
                   'MA_fix_1.0', 'MA_fix_0.9', 'MA_fix_0.8', 'MA_fix_0.7', 'MA_fix_0.6', 'MA_fix_0.5',
                   'MA_decay_0.7', 'MA_decay_0.5',
                   'HA_fix_0.9', 'HA_fix_0.5'
                   ]
    # 'HA_fix_0.9', 'HA_fix_0.8', 'HA_fix_0.7', 'HA_fix_0.6', 'HA_fix_0.5'

    target_dict = {key: target_map_dict[key] for key in target_tags if key in target_map_dict}
    # draw_eval_plot(target_dict=target_dict)

    print_mean_std(target_dict=target_dict, custom_order=target_tags)
