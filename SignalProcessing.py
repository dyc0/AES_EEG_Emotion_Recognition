from xml.dom.expatbuilder import parseFragmentString
import pyxdf
import pandas as pd
from moviepy.editor import VideoFileClip
import numpy as np
from matplotlib import pyplot as plt
import mne

bands = {'delta': [0.5, 4], 'tehta': [4, 8], 'alpha': [8, 13], \
         'beta': [13, 30], 'gamma': [30, 50]}

def getVideoDurations(csvFilePath,
                      videopath='C:\\Users\\Duca\\OneDrive - student.etf.bg.ac.rs\\ETF\\SESTI SEMESTAR\\AES\\PROJEKAT\\VIDEO\\VIDEOS\\'):

    video_names = pd.read_csv(csvFilePath, usecols=['Stimulus'])
    video_indices = [name[0:-4] for name in video_names['Stimulus']]
    video_durations = []
    for name in video_names['Stimulus']:
        video_durations.append(VideoFileClip(videopath + name).duration)
    return video_indices, video_durations




class Segment:

    def __init__(self, raw, videoindex, emotion=None) -> None:
        self.info = raw.info
        self.raw = raw.copy()
        self.videoindex=videoindex
        self.emotion = emotion

        #self.raw.plot(scalings=dict(eeg=100e-6), duration=1, start=14, block=True)
    

    


class EEGData:
    def __init__(self, xdfFilePath, csvFilePath) -> None:
        
        print("Loading data from file...")
        data, header = pyxdf.load_xdf(xdfFilePath)
        self.signal = data[0]['time_series']
        self.timestamps = data[0]['time_stamps']
        self.ch_n = int(data[0]['info']['channel_count'][0])

        chnames = []
        for i in range(self.ch_n):
            chnames.append(data[0]['info']['desc'][0]['channels'][0]['channel'][i]['label'][0])
        print(chnames)
        print("Data loaded.")

        sfreq = int(float(data[0]["info"]["nominal_srate"][0]))
        info = mne.create_info(chnames, sfreq, "eeg")
        self.raw = mne.io.RawArray(self.signal.T * (1e-3 / 50 / 2), info) # CHECK THIS SCALING!!!
        self.raw.set_montage('standard_1020', match_alias=True)
        
        self.raw.plot(scalings=dict(eeg=100e-6), duration=1, start=14, block=True)

        self.segments = []

    def filter_data(self):
        print("Filtering data...")

        filtered_raw = self.raw.copy().filter(1, 40)
        ica = mne.preprocessing.ICA(n_components=self.ch_n-1, max_iter='auto', random_state=97)

        ica.fit(filtered_raw)
        ica.plot_sources(self.raw, block=True)
        ica.plot_components()

        ica.exclude = []
        eog_indices, eog_scores =ica.find_bads_eog(self.raw, ch_name='Fp1', method='correlation', threshold='auto')
        ica.exclude = eog_indices

        #ica.plot_scores(ecg_scores)
        #ica.plot_properties(self.raw, picks=ecg_indices)


        #self.raw = ica.apply(self.raw)
        print("Data filtered.")
    
    def cut_segments(self, indices, durations):
        print('Determining segments timing...')
        indices.reverse()
        durations.reverse()

        currentT = self.raw.times[-1]
        currentT -= 30

        for i in range(len(indices)):
            currentDur = durations[i]
            currentT  -= currentDur
            if currentT < 0:
                break
            self.segments.append(Segment(self.raw.copy().crop(currentT, currentT+currentDur), indices[i]))
            currentT -= 15
            if currentT < 0:
                break
        print('Segments timing determined. ' + str(len(self.segments)) + ' segments created.')

    def assign_markers(self, assignments_path):
        print('Assigning emotion markers...')
        feels = pd.read_csv(assignments_path, usecols=['ID', 'MARKER'])
        for seg in self.segments:
            seg.emotion = feels[feels['ID'] == int(seg.videoindex)]['MARKER'].iloc[0]
        print('Emotion markers assigned.')




if __name__ == '__main__':

    #Plots don't work without this for some reason:
    plt.figure()
    plt.close()

    xdfFilePath = 'xdfs\pesa.xdf'
    csvFilePath = 'csvs\pesa.csv'

    eeg = EEGData(xdfFilePath, csvFilePath)
    vidind, viddur = getVideoDurations(csvFilePath)
    eeg.filter_data()
    #eeg.cut_segments(vidind, viddur)
    #eeg.assign_markers('csvs\\video_info.csv')

    '''features = pd.DataFrame()
    ind = 0
    for feature in eeg.segments:
        features[ind] = feature.get_features_flat()
        ind+=1
    features.set_axis(eeg.segments[0].get_axis_names(), axis='index', inplace=True)
    features = features.T   
        
    features.to_csv('Features.csv')'''

    print('DONE.')
