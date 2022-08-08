import timer

import pyxdf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

import mne

from sklearn import decomposition

bands = {'delta': [0.5, 4], 'tehta': [4, 8], 'alpha': [8, 13], \
         'beta': [13, 30], 'gamma': [30, 50]}


class Emotion:
    def __init__(self, csv_evals=None, csv_evals_path=None) -> None:
        if not csv_evals:
            self.cf = pd.read_csv(csv_evals_path)
        else:
            self.cf = csv_evals
    
    def get_class(self, vidindex):
        return self.cf[self.cf['ID'] == vidindex]['MARKER'].iloc[0]
        


class Segment:

    def __init__(self, data, index, freq, emotion: Emotion) -> None:
        self.data = data
        self.index = index
        self.emotion = emotion.get_class(int(self.index))
        self.freq = freq

        ica = decomposition.FastICA(n_components=22, whiten='arbitrary-variance', random_state=97)
        self.filtered_data = ica.fit_transform(self.data)

        self.calculate_mean()
        self.calculate_stdev()
        self.calculate_minmax()

        self.get_ffts()
        self.get_band_stats()

        self.power = 0
        self.energy = 0

    def get_ffts(self) -> None:
        self.FFTdata = []
        for column in self.filtered_data.T:
            self.FFTdata.append(np.fft.fft(column))
        self.FFTdata = np.transpose(self.FFTdata)

    def get_band_stats(self):
        self.channel_band_stats = []
        for column in self.FFTdata.T:
            band_stats = {}
            for key in bands.keys():
                valid_data = np.absolute(column[int(bands[key][0]*self.freq):int(bands[key][1]*self.freq)])
                min = np.min(valid_data)
                max = np.max(valid_data)
                mean = np.mean(valid_data)
                stdev = np.std(valid_data)
                pow = sum(i*i for i in valid_data)
                band_stats[key] = [min, max, mean, stdev, pow]
            self.channel_band_stats.append(band_stats)
        

    def calculate_mean(self) -> None:
        self.mean = []
        for column in range(self.filtered_data.shape[1]):
            self.mean.append(np.mean(self.filtered_data[:, column]))

    def calculate_minmax(self) -> None:
        self.min =[]
        self.max = []
        for column in range(self.filtered_data.shape[1]):
            self.min.append(min(self.filtered_data[:, column]))
            self.max.append(max(self.filtered_data[:, column]))

    def calculate_stdev(self) -> None:
        self.stdev = []
        for column in range(self.filtered_data.shape[1]):
            self.stdev.append(np.std(self.filtered_data[:, column]))

    def get_features_flat(self):
        array = []
        array.extend(self.min)
        array.extend(self.max)
        array.extend(self.mean)
        array.extend(self.stdev)
        for channel in range(self.data.shape[1]):
            for key in bands.keys():
                array.extend(self.channel_band_stats[channel][key])
        array.append(self.emotion)
        return array
    
    def get_axis_names(self):
        array = []
        for i in range(self.filtered_data.shape[1]):
            array.append("CHL_" + str(i) + "_MIN")
        for i in range(self.filtered_data.shape[1]):
            array.append("CHL_" + str(i) + "_MAX")
        for i in range(self.filtered_data.shape[1]):
            array.append("CHL_" + str(i) + "_MED")
        for i in range(self.filtered_data.shape[1]):
            array.append("CHL_" + str(i) + "_STD")
        for i in range(self.filtered_data.shape[1]):
            for key in bands.keys():
                name = "CHL_" + str(i) + "_" + key + "_"
                array.append(name + "MIN")
                array.append(name + "MAX")
                array.append(name + "MED")
                array.append(name + "STD")
                array.append(name + "POW")
        array.append('EMOTION')
        return array



    def __str__(self) -> str:
        rtrStr = "Features for interval no. " + str(self.index) + ":\n"
        rtrStr += "mean: " + str(self.mean) + "\n"
        rtrStr += "std: " + str(self.stdev) + "\n"
        
        return rtrStr


class EEGData:

    def __init__(self, xdfFilePath, csvFilePath, emotion: Emotion,
              videopath='C:\\Users\\Duca\\OneDrive - student.etf.bg.ac.rs\\ETF\\SESTI SEMESTAR\\AES\\PROJEKAT\\VIDEO\\VIDEOS\\'):
        
        self.emotion = emotion

        self.data, self.header = pyxdf.load_xdf(xdfFilePath)
        self.signal = self.data[0]['time_series']
        self.timestamps = self.data[0]['time_stamps']
        self.freq = float(self.data[0]["info"]["nominal_srate"][0])

        timeIndices, video_indices, video_durations = timer.loadTimes(xdfFilePath, csvFilePath)
        self.segments = timer.calculateSegments(timeIndices, video_indices, video_durations , self.freq)

        self.extract_data()


    def extract_data(self, offset=20):
        self.intervals = {}
        
        for segment in self.segments:
            startI = segment.startIndex + int(offset*self.freq)
            endI = segment.endIndex - int(offset*self.freq)
            self.intervals[segment.index] = self.signal[startI:endI, :]


    def extract_segments(self):
        self.features = []
        for segment in self.segments:
            self.features.append(Segment(self.intervals[segment.index], segment.index, self.freq, self.emotion))



class Features:
    def __init__(self, eeg_data) -> None:
        self.features = pd.DataFrame()
        for feature in eeg_data.features:
            self.features[feature.index] = feature.get_features_flat()
        self.features.set_axis(eeg_data.features[0].get_axis_names(), axis='index', inplace=True)
        self.features = self.features.T   
        

    def to_csv(self):
        self.features.to_csv('Features.csv')

if __name__ == '__main__':
    emo = Emotion(csv_evals_path='csvs\\video_info.csv')
    eeg = EEGData('xdfs\pesa.xdf', 'csvs\pesa.csv', emo)
    eeg.extract_data()
    eeg.extract_segments()
    fea = Features(eeg)
    fea.to_csv()
    print('Finished.')