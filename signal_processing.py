import timer

import pyxdf
import matplotlib.pyplot as plt
import numpy as np


class Features:

    def __init__(self, data, index) -> None:
        self.data = data
        self.index = index

        self.calculate_mean()
        self.calculate_stdev()
        self.power = 0
        self.minimum = 0
        self.maximum = 0
        self.energy = 0

    def calculate_mean(self):
        self.mean = []
        for column in range(self.data.shape[1]):
            self.mean.append(np.mean(self.data[:, column]))

    def calculate_stdev(self):
        self.stdev = []
        for column in range(self.data.shape[1]):
            self.stdev.append(np.std(self.data[:, column]))

    def __str__(self) -> str:
        rtrStr = "Features for interval no. " + str(self.index) + ":\n"
        rtrStr += "mean: " + str(self.mean) + "\n"
        rtrStr += "std: " + str(self.stdev) + "\n"
        
        return rtrStr




class EEGData:

    def __init__(self, xdfFilePath, csvFilePath,
              videopath='C:\\Users\\Duca\\OneDrive - student.etf.bg.ac.rs\\ETF\\SESTI SEMESTAR\\AES\\PROJEKAT\\VIDEO\\VIDEOS\\'):
        
        self.data, self.header = pyxdf.load_xdf(xdfFilePath)
        self.signal = self.data[0]['time_series']
        self.timestamps = self.data[0]['time_stamps']

        timeIndices, video_indices, video_durations = timer.loadTimes(xdfFilePath, csvFilePath)
        self.segments = timer.calculateSegments(timeIndices, video_indices, video_durations)

        self.extract_data()


    def extract_data(self, offset=20, freq=250):
        self.intervals = {}
        
        for segment in self.segments:
            startI = segment.startIndex + offset*freq
            endI = segment.endIndex - offset*freq
            self.intervals[segment.index] = self.signal[startI:endI, :]


    def extract_features(self):
        self.features = []
        for segment in self.segments:
            self.features.append(Features(self.intervals[segment.index], segment.index))
        
        for ftr in self.features:
            print(ftr)



if __name__ == '__main__':
    dataframe = EEGData('xdfs\pesa.xdf', 'csvs\pesa.csv')
    dataframe.extract_data()
    dataframe.extract_features()