import timer

import pyxdf
import matplotlib.pyplot as plt
import numpy as np

import mne

class Segment:

    def __init__(self, signal, index, freq) -> None:
        self.index = index

        self.info = mne.create_info(22, freq, "eeg")
        raw = mne.io.RawArray(signal.T * (1e-3 / 50 / 2), self.info) # CHECK THIS SCALING!!!
        raw.plot(scalings=dict(eeg=100e-6), duration=1, start=14, block=True)
        ica = mne.preprocessing.ICA(n_components=21, max_iter='auto', random_state=97)
        ica.fit(raw)
        ica
        ica.plot_sources(raw, show_scrollbars=False)

class EEGData:

    def __init__(self, xdfFilePath, csvFilePath,
              videopath='C:\\Users\\Duca\\OneDrive - student.etf.bg.ac.rs\\ETF\\SESTI SEMESTAR\\AES\\PROJEKAT\\VIDEO\\VIDEOS\\'):
        
        self.data, self.header = pyxdf.load_xdf(xdfFilePath)
        self.signal = self.data[0]['time_series']
        self.timestamps = self.data[0]['time_stamps']
        self.freq = float(self.data[0]["info"]["nominal_srate"][0])

        timeIndices, video_indices, video_durations = timer.loadTimes(xdfFilePath, csvFilePath)
        self.segment_times = timer.calculateSegments(timeIndices, video_indices, video_durations)

        self.extract_data()


    def extract_data(self, offset=20):
        self.intervals = {}
        
        for segment in self.segment_times:
            startI = segment.startIndex + offset*int(self.freq)
            endI = segment.endIndex - offset*int(self.freq)
            self.intervals[segment.index] = self.signal[startI:endI, :]


    def extract_features(self):
        self.segments = []
        for segment in self.segment_times:
            self.segments.append(Segment(self.intervals[segment.index], segment.index, self.freq))


if __name__ == '__main__':
    dataframe = EEGData('xdfs\LAZAR.xdf', 'csvs\LAZAR.csv')
    dataframe.extract_data()
    dataframe.extract_features()