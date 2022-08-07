import pyxdf
import pandas as pd
from moviepy.editor import VideoFileClip


class Interval:

    def __init__(self, start=0, end=0, videoIndex=0):
        self.startIndex = start
        self.endIndex = end
        self.index = videoIndex

    def __str__(self):
        return "start: " + str(self.startIndex) + "\nend: " + \
                str(self.endIndex) + "\nindex: " + str(self.index)


def loadTimes(xdfFilePath, csvFilePath,
              videopath='C:\\Users\\Duca\\OneDrive - student.etf.bg.ac.rs\\ETF\\SESTI SEMESTAR\\AES\\PROJEKAT\\VIDEO\\VIDEOS\\'):
    data, header = pyxdf.load_xdf(xdfFilePath)

    video_names = pd.read_csv(csvFilePath, usecols=['Stimulus'])
    video_indices = [name[0:-4] for name in video_names['Stimulus']]
    video_durations = {}
    for name in video_names['Stimulus']:
        video_durations[name[0:-4]] = VideoFileClip(videopath + name).duration

    return len(data[0]['time_stamps']), video_indices, video_durations

    
def calculateSegments(time, video_indices, video_durations, freq=250):
    freq = int(freq)
    currentIndex = time - 30*freq
    intervals = []

    for i in reversed(video_indices):
        videoSamples = int(video_durations[i]*freq)

        inter = Interval()
        inter.index = i
        inter.endIndex = currentIndex;
        currentIndex -= videoSamples
        if currentIndex < 0:
            break
        inter.startIndex = currentIndex
        currentIndex -= 15*freq
        intervals.append(inter)
        if currentIndex < 0:
            break

    return intervals


if __name__=='__main__':
    xdfFile = 'xdfs\pesa.xdf'
    csvFile = 'csvs\pesa.csv'

    time, video_indices, video_durations = loadTimes(xdfFile, csvFile)
    calculateSegments(time, video_indices, video_durations)