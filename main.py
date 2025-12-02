'''

Module: COMPACT

Author: Gian Paolo Marra    <gianpaolo.marra@cnr.it>
        Stefano Pascali     <stefano.pascali@cnr.it>
        Gabriele Surano     <gabriele.surano@cnr.it>

Author URI: https:

Python Version: 3.10

Tested up to: 3.10

Requires:   requirements.txt
    datatime, numpy, pandas, cv2, sys, csv, time, scipy.spatial, matplotlib.pyplot,
    seaborn, sklearn.linear_model

License: GNU General Public License v2 or later

License URI: http://www.gnu.org/licenses/gpl-2.0.html

Description:

    [EN]
    The system starts the analysis as soon as it detects the movement and
    takes measurements of the position of the movement in every specific frame,
    defined by the sampling interval. At the end of the acquisition,
    the software performs a further analysis calculating the distance traveled
    between one observation and the next both in px and in nm.
    A third outliers correction analysis is carried out in the second instance.
    After defining confidence threshold values that are not within the range
    of the identified distribution they take the value of the nearest limit threshold (higher or lower).
    The system returns a graph with the distance traveled.

    [IT]
    Il sistema inizia l'analisi appena rilava del movimento ed
    effettua delle rilevazioni sulla posizione del movimento ogni
    frame specifico, definito dall'intervallo di campionamento.
    Al termine dall'acquisizione, il software effettua un ulteriore analisi
    calcolando la distanza percorsa tra un'osservazione e quella successiva
    sia in px che in nm. Una terza analizi di correzione degli outliers viene
    effettuata in seconda battuta. Dopo aver definito una soglia di confidenza
    i valori non compresi nell'intervallo della distribuzione individuata
    assumono il valore del limite-soglia più vicino (superio o inferiore).
    Il sistema restituisce un grafico con la distanza percorsa.

Tags: motion detection, distance calculation, speed calculation by fps

'''

import os
import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
from numpy import genfromtxt
import cv2
import sys
import csv
import time
from scipy.spatial import distance
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import entropy


sns.set( style = "dark", color_codes = True )

TEXT_COLOR = (0, 255, 0)
TEXT_INFO_COLOR = (255, 255, 255)
TRACKER_COLOR = (255, 0, 0)
FONT = cv2.FONT_HERSHEY_SIMPLEX

VIDEO_SOURCE = "video/20251015_154007768-15fps-x20RTL.mov"

# VIDEO_SOURCE = "video/20251015_154007768-15fps-x20RTL.mov"
# VIDEO_SOURCE = "video/20241022_172649156_5ul_20glic-x20RTL.mov"
# VIDEO_SOURCE = "video/20241023_174413659_5ul_0glic-x20RTL.mov"


BGS_TYPES = [ "GMG", "MOG", "MOG2", "KNN", "CNT" ]
BGS_TYPE = BGS_TYPES[ 2 ]

class fluid_displacement_detection:

    def __init__(self, video="", title=""):

        self.direzione = -1  # > 0 if left-to-right   or     < 0 if right-to-left
        self.sampling_time = 1  # sampling time in seconds

        self.magnification_rate = 0
        self.real_width = 0   # actual image width in mm
        self.real_height = 0  # actual image height in mm
        self.field_depth = 0
        self.work_distance = 0
        # Field of view at 50x zoom --> 7.8mm x 5.8mm.

        self.zoom = 0
        self.scale_factor = 1  # scale video in percentage -> displacement calculation (x,y, nframe ) -> dist.euclidea in px
        self.mm_to_px = 1 / 320  # Scale factor for converting from px to nm

        self.optical_data_table = "optical_data_table/dino-lite.csv"

        ### DEFAULT A 0 - INIT AFTER START ###

        self.video_width = 0  # width of frames in px
        self.video_height = 0  # height of frames in px
        self.video_height_cut = 0.6  # portion of the height to be retained after cutting
        self.revision = '1. 28-10-24'  # revision nr.
        self.tot_frame = 0  # total number of frames detected in the video
        self.durata = 0  # video duration in seconds
        self.title = title  # name of experiment
        self.video = video  # name of experiment video
        self.minArea = 250  # minimum threshold of moving area to be detected in px
        self.maxArea = self.minArea + 1000000000

        self.medianBlur = 5  # blur on analysis mask [0,10] x10
        self.soglia = 0.08  # threshold for defining outliers from the regr. estimated on the data (with outliers)
        self.scost = 0
        self.fps = -15

        self.show_result = True
        self.show_mask = False
        self.show_cropped = False

        self.verbose = False

        self.mean_greys = []
        self.var_greys = []
        self.contrast = []
        self.entropy = []
        self.laplace_variance = []

    def start(self):

        ltr = 0  # flow direction (0 if right-to-left; 1 if left-to-right)
        direction_prev = 0  # direction of the previous bearing
        self.direzione_movimento()
        self.getZoom()

        nrel = 0  # number detection
        rilevamenti = []  # detections to save ( nFrame )
        sampling_counter = 0

        time_start = 0  # movement start time
        time_end = 0  # movement end time

        current_frame = 0  # frame current
        frame_start = 0  # first frame where the movement started
        frame_end = 0  # last frame where motion was detected
        frame_prev = 0

        self.logIt(f"Experiment: {self.title}", "w")
        self.logIt(f"Start video analysis: {self.video}")
        self.logIt(f"Algorithm revision: {self.revision}")

        self.logIt(f"Movement direction: {self.direzione}")
        self.logIt(f"Zoom: {self.zoom}")

        self.logIt(f"Field of view: {self.real_width} x {self.real_height} mm")
        self.logIt(f"Depth of field: {self.field_depth} mm")
        self.logIt(f"Working distance of the microscope: {self.work_distance} mm")

        cap = cv2.VideoCapture(self.video)
        bg_subtractor = self.getBGSubtractor(BGS_TYPE)

        while (cap.isOpened):

            ok, frame = cap.read()

            if not ok:
                self.logIt("Video processing finished.")
                break

            current_frame += 1

            # frame = cv2.resize(frame, (0, 0), fx=0.50, fy=0.50)
            # frame = cv2.resize(frame, (0, 0), fx=self.scale_factor, fy=self.scale_factor)

            video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cut_padding = int((video_height * self.video_height_cut) / 2)
            frame = frame[cut_padding: -cut_padding].copy()

            if current_frame == 1:

                # get information about the video
                fps = round( cap.get( cv2.CAP_PROP_FPS ), 2 )  # OpenCV v2.x used "CV_CAP_PROP_FPS"
                video_width = int( cap.get( cv2.CAP_PROP_FRAME_WIDTH ))
                video_height = int( cap.get( cv2.CAP_PROP_FRAME_HEIGHT ))
                tot_frame = int( cap.get( cv2.CAP_PROP_FRAME_COUNT ))
                durata = round( tot_frame / fps, 2 )
                minuti = int( durata / 60)
                secondi = round( durata % 60, 2 )

                self.video_width = video_width
                self.video_height = video_height
                self.tot_frame = tot_frame
                self.durata = durata
                self.fps = fps

                self.logIt(f"Dim. frame (px): {video_width} x {video_height}")
                self.logIt(f"Dim. frame (mm): {self.real_width} x {self.real_height}")
                self.logIt('Frame Rate (fps): ' + str(fps))
                self.logIt('Tot. number of frames in the video: ' + str(tot_frame))
                self.logIt('Video length (sec): ' + str(durata))
                self.logIt('Video length (m:s): ' + str(minuti) + ':' + str(secondi))
                self.logIt('Sampling interval (sec): ' + str(self.sampling_time))

                self.logIt("Edge trimming factor (top and bottom): " + str(self.video_height_cut))
                self.logIt("Px cut from the frame: " + str(cut_padding) + "px")
                self.logIt("Frame size after cropping: " + str(np.shape(frame)))

                f_height, f_width, color = frame.shape
                out = cv2.VideoWriter( f"{self.title}/processed_video.avi", cv2.VideoWriter_fourcc(*'MJPG'), fps, (f_width, f_height))


            # apply morphological filters of the 'combine' type to the current frame
            bg_mask = bg_subtractor.apply( frame )
            bg_mask = self.getFilter( bg_mask, 'combine')
            # bg_mask = cv2.medianBlur(bg_mask, self.medianBlur)
            bg_mask = cv2.GaussianBlur( bg_mask, (5, 5), 0)

            (contours, hierarchy) = cv2.findContours( bg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE )
            # print(contours)

            self.lighting_analyze(frame)

            video_height = np.shape(frame)[0]

            cv2.rectangle( frame, (0, 0), (video_width, 25), (0, 0, 0), -1)
            cv2.rectangle( frame, (0, int(video_height * self.scale_factor) - 30),
                          ( video_width, int( video_height * self.scale_factor )), (0, 0, 0), -1)

            for cnt in contours:
                area = cv2.contourArea( cnt )

                # check if the detected movement is within the expected range
                if area >= self.minArea and area <= self.maxArea and frame_prev != current_frame:

                    x, y, w, h = cv2.boundingRect(cnt)
                    # cv2.rectangle(frame, (10,30), (250,55), (255,0,0), -1)

                    # except the frame number of the first motion detection
                    if (nrel == 1):
                        frame_start = current_frame
                        time_start = time.time()
                        direction_prev = x

                    # except the frame number of the last motion detection
                    frame_end = current_frame
                    time_end = time.time()

                    # ccalculate the duration of the entire movement
                    mov_duration = round(float(time_end - time_start), 2)

                    # top left print: first frame no., last frame no., movement duration
                    cv2.putText(frame, f'1st frame: {frame_start}', (200, 18), FONT, 0.6, TEXT_COLOR, 1, cv2.LINE_AA)
                    cv2.putText(frame, f'last frame: {frame_end}', (400, 18), FONT, 0.6, TEXT_COLOR, 1, cv2.LINE_AA)
                    cv2.putText(frame, f'Durata: ' + str(mov_duration) + "sec.", (600, 18), FONT, 0.6, TEXT_COLOR, 1,
                                cv2.LINE_AA)

                    # bottom right mold: the sample number
                    # cv2.putText(frame, f'n.campione: ' + str( len( rilevamenti )), (200, int( video_height * scale_factor ) - 10 ), FONT, 0.6, TEXT_COLOR, 1, cv2.LINE_AA)
                    # cv2.rectangle( frame, (direction_prev, 26 ), (direction_prev + 3, int( video_height * scale_factor ) - 30 ), (0, 0, 255), -1)

                    # print(f"{x} - {direction_prev} = " + str(x - direction_prev))

                    # check the direction of the flow, comparing the previous x
                    if ltr == 0 and x - direction_prev < 0:
                        direction_prev = x

                    # except for the findings made by the video
                    rilevamenti += [{"n": len(rilevamenti) + 1,
                                     "x": x, "y": y,
                                     "f": current_frame,
                                     "mean_greys": self.mean_greys[-1],
                                     "var_greys": self.var_greys[-1],
                                     "contrast": self.contrast[-1],
                                     "entropy": self.entropy[-1],
                                     "laplace_variance": 0
                                     }]

                    # print( str(sampling_time * fps ) + " - " + str( current_frame ))

                    sampling_counter = 0

                    # highlights area where motion was detected
                    cv2.drawContours(frame, cnt, -1, TRACKER_COLOR, 3)
                    cv2.drawContours(frame, cnt, -1, (255, 255, 255), 1)
                    cv2.rectangle(frame, (x, y), (x + w, y + h), TRACKER_COLOR, 3)
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 255), 1)

                    # draws the outline of the pixels that have changed since the previous frame
                    for alpha in np.arange(0.8, 1.1, 0.9)[::-1]:
                        frame_copy = frame.copy()
                        output = frame.copy()
                        cv2.drawContours(frame_copy, [cnt], -1, TRACKER_COLOR, -1)
                        frame = cv2.addWeighted(frame_copy, alpha, output, 1 - alpha, 0, output)

                    frame_prev = current_frame
                    nrel += 1

            sampling_counter += 1

            # top left corner: print the current frame number
            cv2.putText(frame, f'cur.frame: {current_frame}', (10, 18), FONT, 0.6, TEXT_INFO_COLOR, 1, cv2.LINE_AA)

            # bottom left corner: print sampling interval and number of samples
            cv2.putText(frame, f'interv.camp.: All', (10, int(video_height * self.scale_factor) - 11), FONT, 0.6,
                        TEXT_INFO_COLOR, 1, cv2.LINE_AA)
            cv2.putText(frame, f'Field of view.: {self.real_width}x{self.real_height}mm',
                        (200, int(video_height * self.scale_factor) - 11), FONT, 0.6, TEXT_INFO_COLOR, 1, cv2.LINE_AA)
            cv2.putText(frame, f'Magnification: {self.zoom}', (500, int(video_height * self.scale_factor) - 11), FONT,
                        0.6, TEXT_INFO_COLOR, 1, cv2.LINE_AA)
            cv2.putText(frame, f'Algorithm rev.: {self.revision}', (750, int(video_height * self.scale_factor) - 11),
                        FONT, 0.6, TEXT_INFO_COLOR, 1, cv2.LINE_AA)

            result = cv2.bitwise_and( frame, frame, mask = bg_mask )

            if self.show_result:
                cv2.imshow('Frame', frame)

            if self.show_mask:
                cv2.imshow('Mask', result)

            if self.show_cropped:
                crop_y = 375
                crop_h = 430
                crop_x = 0
                crop_w = video_width
                crop_image = result[crop_y: crop_y + crop_h, crop_x: crop_x + crop_w]
                cv2.imshow("cropped", crop_image)

            out.write(frame)

            # If the user presses the 'q' key, the video analysis stops.
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        # end of analysis loop
        ################################################################

        self.logIt(f"Frame where the movement started: " + str(frame_start))
        self.logIt(f"Frame where the movement ended: " + str(frame_end))
        self.logIt(f"Number of surveys carried out: " + str(len(rilevamenti)) + " / " + str(nrel))
        self.logIt(f"Movement duration calculated on processing timestamp (sec): " + str(
            round(time_end - time_start, 2)))
        self.logIt(f"Motion duration calculated on fps: " + str(round((frame_end - frame_start) / fps, 2)))
        self.logIt(f"Medium intensity grays: " + str(np.mean(self.mean_greys)))
        self.logIt(f"Gray intensity variance: " + str(np.mean(self.var_greys)))
        self.logIt(f"Contrast: " + str(np.mean(self.contrast)))
        #self.logIt(f"Shannon Entropy: " + str(np.mean(self.entropy)) + " [0,8] for a 256bit image -> log^2(256)=8")
        self.logIt(f"Gradient variance/Laplacian: " + str(np.mean(self.laplace_variance)))

        # write the data to a csv file
        self.salva_rilevamenti( rilevamenti, direzione = direction_prev )
        self.visualizza_grafico()



    def lighting_analyze(self, frame):

        frame = frame.astype(np.float64)

        # get the mean and variance of the grayscale pixel intensity
        mean_greys = np.mean(frame)
        self.mean_greys.append(mean_greys)

        var_greys = np.var(frame)
        self.var_greys.append(var_greys)

        # get the contrast
        contrast = int((frame.max() - frame.min()) / (frame.max() + frame.min()))
        self.contrast.append(contrast)

        hist = np.histogram( frame, bins=256, range=(0, 255))[0]
        prob = hist / np.sum(hist)
        prob = prob[prob > 0]  # Eliminate null probabilities to avoid nan in the log
        entropy_value = entropy( prob, base=2)

        # entropy = scipy.stats.entropy( frame )
        self.entropy.append(entropy_value)

        # laplace_variance = scipy.stats.entropy( np.histogram( frame , bins = 256 ))
        laplace_variance = scipy.stats.entropy(frame)
        self.laplace_variance.append(laplace_variance)

    def visualizza_grafico( self, filePath='automated.csv' ):

        headerKey = ['n', 'f', 'x_px', 'y_px', 'dist_px',
                     'x_mm', 'y_mm', 'dist_mm',
                     'scost_medio_mm', 'mean_greys', 'var_greys',
                     'contrast', 'entropy', 'laplace_variance']

        # visualizzo il grafico in mathplot
        data = genfromtxt(f"{self.title}/{filePath}", delimiter=',', names=headerKey)

        df = pd.DataFrame(data, columns=data.dtype.names)

        # Remove rows with NaN values
        df_clean = df.dropna()

        # View Results
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(df_clean['n'], df_clean['x_mm'], cmap='viridis', s=15, label="Observations")
        plt.title('Motion detection of a meniscus')
        plt.xlabel('Time (in frames)')
        plt.ylabel('Distance (nm)')

        frames = int(self.tot_frame)
        fps = int(self.fps)

        # 10s time ruler
        for i in range(1, frames, fps):

            if int(i / fps) == 0:
                plt.plot([i, i], [-0.1, -0.25], color="red", label="Seconds")
            if int(i / fps) % 10 == 0:
                plt.plot([i, i], [-0.1, -0.25], color="red")
            elif int(i / fps) % 5 == 0:
                plt.plot([i, i], [-0.1, -0.20], color="red")
            else:
                plt.plot([i, i], [-0.1, -0.15], color="red")

        plt.legend()
        plt.grid()
        plt.show()



    def salva_rilevamenti( self, rilevamenti = [], direzione = 1, filePath = 'automated.csv' ):

        w_fact = self.real_width / self.video_width
        h_fact = self.real_width / self.video_height

        self.logIt(f"Conversion factors from px to nm: ( x, y ) -> ({w_fact} , {h_fact} )")

        self.logIt("Start saving the survey file.")

        # headerKey = ['n', 'x', 'y', 'f', 'dist', 'scost_medio'] mean_greys
        headerKey = ['n', 'f', 'x_px', 'y_px', 'dist_px',
                     'x_mm', 'y_mm', 'dist_mm',
                     'scost_medio_mm', 'mean_greys', 'var_greys',
                     'contrast', 'entropy', 'laplace_variance'
                     ]

        nrow = 0
        prev_row = rilevamenti[0]
        prev_dist_mm = 0
        dist_mm = 0

        kx = [i['x'] for i in rilevamenti if 'x' in i]
        print(kx)

        if self.direzione == "rtl":
            # kx.reverse()
            # print(rilevamenti[nrow]['x'])
            for i in range(len(rilevamenti)):
                rilevamenti[i]['x'] = self.video_width - kx[i]

            self.logIt(f"Movement from the right --> I rotate the coordinates")
            print( rilevamenti[i]['x'] )
            # df_clean.loc[:, 'x_mm'] = df_clean['x_mm'].values[::-1]

        with open(f"{self.title}/{filePath}", "w") as f:

            headVal = csv.DictWriter(f, fieldnames=headerKey)
            headVal.writeheader()
            nrow = 0

            for k in rilevamenti:

                if nrow > 0:

                    # getting the coordinates of the start and end point in px
                    # and calculating the distance in px from the last point
                    startP = ( prev_row['x'], prev_row['y'] )
                    endP = ( k['x'], k['y'] )
                    dist_px = distance.euclidean( startP, endP )

                    # convert coordinates from px to mm
                    startP = (prev_row['x'] * w_fact, prev_row['y'] * h_fact)
                    endP = (k['x'] * w_fact, k['y'] * h_fact)
                    dist_mm = distance.euclidean(startP, endP)

                    # calculate the average deviation from the last observation
                    if nrow == 0:
                        scost_medio = 0
                    else:
                        scost_medio = np.mean(( startP[0], endP[0] ))
                        # scost_medio = np.mean(( prev_dist_mm, dist_mm ))

                    datas = {'n': k['n'],
                             'f': k['f'],
                             'x_px': k['x'],
                             'y_px': k['y'],
                             'dist_px': dist_px,
                             'x_mm': k['x'] * w_fact,
                             'y_mm': k['y'] * h_fact,
                             'dist_mm': dist_mm,
                             'scost_medio_mm': scost_medio,
                             'mean_greys': k['mean_greys'],
                             'var_greys': k['var_greys'],
                             'contrast': k['contrast'],
                             'entropy': k['entropy'],
                             'laplace_variance': k['laplace_variance']}

                    # print( datas )
                    headVal.writerow(datas)

                prev_row = k
                prev_dist_mm += dist_mm
                nrow += 1

        f.close()
        self.logIt("End of saving of survey files.")

    def logIt(self, text="", mod="a"):

        with open(f"{self.title}/log.csv", mod) as f:
            headVal = csv.DictWriter(f, fieldnames = [ 'time', 'info' ])

            if mod == "w":
                headVal.writeheader()

            datas = {'time': datetime.datetime.now(), 'info': text}

            headVal.writerow(datas)
            print(text)

    def direzione_movimento(self):

        # looking for the substring rtl (Right-To-Left)
        # or ltr (Left-To-Right) in filename

        if "RTL" in self.video or "rtl" in self.video:
            self.direzione = "rtl"
        elif "LTR" in self.video or "ltr" in self.video:
            self.direzione = "ltr"
        else:
            self.direzione = 0

        # > 0 se ltr
        # < 0 se rtl
        # = 0 se ND
        return self.direzione



    def getZoom(self):

        if "-x" in self.video:

            zoom_pos = self.video.find("-x")
            self.zoom = self.video[ zoom_pos + 1: zoom_pos + 4 ]
            self.getRealDimension()

            return self.zoom
        else:
            return 0



    def getRealDimension(self):

        z = int(self.zoom[1:])

        if z > 0:

            df = pd.read_csv( self.optical_data_table , header=0, sep=',', decimal=',')

            self.magnification_rate = df.magn_rate[df["magn_rate"] == z].values[0]
            self.real_width = df.field_view_x[df["magn_rate"] == z].values[0]
            self.real_height = df.field_view_y[df["magn_rate"] == z].values[0]
            self.field_depth = df.field_depth[df["magn_rate"] == z].values[0]
            self.work_distance = df.work_dist[df["magn_rate"] == z].values[0]

            return True
        else:
            return False



    def getKernel( self, KERNEL_TYPE ):

        if KERNEL_TYPE == "dilation":
            kernel = cv2.getStructuringElement( cv2.MORPH_ELLIPSE, ( 3, 3 ))
        if KERNEL_TYPE == "opening":
            kernel = np.ones(( 3, 3 ), np.uint8 )
        if KERNEL_TYPE == "closing":
            kernel = np.ones(( 3, 3 ), np.uint8 )

        return kernel



    def getFilter( self, img, filter ):

        if filter == 'closing':
            return cv2.morphologyEx(img, cv2.MORPH_CLOSE, self.getKernel("closing"), iterations=2)

        if filter == 'opening':
            return cv2.morphologyEx(img, cv2.MORPH_OPEN, self.getKernel("opening"), iterations=2)

        if filter == 'dilation':
            return cv2.dilate(img, self.getKernel("dilation"), iterations=2)

        if filter == 'combine':
            closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, self.getKernel("closing"), iterations=2)
            opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, self.getKernel("opening"), iterations=2)
            dilation = cv2.dilate(opening, self.getKernel("dilation"), iterations=2)

            return dilation



    def getBGSubtractor( self, BGS_TYPE ):
        if BGS_TYPE == "GMG":
            return cv2.bgsegm.createBackgroundSubtractorGMG()
        if BGS_TYPE == "MOG":
            return cv2.bgsegm.createBackgroundSubtractorMOG()
        if BGS_TYPE == "MOG2":
            return cv2.createBackgroundSubtractorMOG2()
        if BGS_TYPE == "KNN":
            return cv2.createBackgroundSubtractorKNN()
        if BGS_TYPE == "CNT":
            return cv2.bgsegm.createBackgroundSubtractorCNT()
        print("Invalid detector")
        sys.exit(1)



# create a folder to save the analysis outputs with
# the same name of video and in the same location

video_exp = VIDEO_SOURCE
experiment_name = os.path.splitext( os.path.basename( video_exp ))[0]

file_path = Path( video_exp )
folder_exp = file_path.parent / experiment_name

if not folder_exp.exists():
    folder_exp.mkdir()

experiment_name = os.path.dirname( video_exp ) + '/' + experiment_name



# start the analysis
COMPACT = fluid_displacement_detection( video_exp, experiment_name )
COMPACT.start()
