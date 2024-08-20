import os, glob
import pandas as pd
import matplotlib.pyplot as plt


class SmoothedPlot:
    def __init__(self, smoothing_factor=0.6, root_path='data', target_ext='csv'):
        self.root_path = root_path
        self.target_ext = target_ext
        self.smoothing_factor = smoothing_factor

        self.raw_label = False

        self.data_list = []
        self.labels = []

    def fetch_files_from_root(self):
        _current_directory = os.path.dirname(os.path.abspath(__file__))
        current_directory = os.path.join(_current_directory, self.root_path)

        search_pattern = os.path.join(current_directory, f'*.{self.target_ext}')
        files = glob.glob(search_pattern)
        return files

    def load_data(self, file_path, label=None):
        data = pd.read_csv(file_path)
        smoothed_values = self.smooth_values(data['Value'])
        self.data_list.append((data['Step'], data['Value'], smoothed_values))
        if label is None:
            label = file_path.split('/')[-1]
        self.labels.append(label)

    def smooth_values(self, values):
        smoothed = []
        for i in range(len(values)):
            if i == 0:
                smoothed.append(values[i])
            else:
                smoothed_value = smoothed[-1] * (1 - self.smoothing_factor) + values[i] * self.smoothing_factor
                smoothed.append(smoothed_value)
        return smoothed

    def plot(self):
        plt.figure(figsize=(10, 6))

        for (step, raw_values, smoothed_values), label in zip(self.data_list, self.labels):
            # Raw data (faint)
            plt.plot(step, raw_values, alpha=0.3, label=f'Raw {label}' if self.raw_label else None)
            # Smoothed data (vivid)
            plt.plot(step, smoothed_values, label=f'{label}')

        plt.title('Bimanual Throw/Catch Reward Plot')
        plt.xlabel('Epoch')
        plt.ylabel('Reward')
        plt.grid(True)
        plt.legend()
        plt.show()


if __name__ == '__main__':
    plotter = SmoothedPlot(smoothing_factor=0.6)

    files = plotter.fetch_files_from_root()
    print("files: ", files)
    for file in files:
        file_name = file.split('/')[-1]
        label = file_name.split('.')[-2]
        plotter.load_data(file, label)

    plotter.plot()
