import json
import os
from PIL import Image
import pickle
import numpy as np
import cv2
import scipy
from scipy import ndimage
import random
import auto_align
import getLines
import matplotlib.pyplot as plt
import pickle
import random
import numpy as np

def getLatestTrack(idx = -1):
    dir_ = "../CameraMotionLogger/Server/website/uploads/"
    #file = sorted(os.listdir(dir_))[-10]
    file = sorted(os.listdir(dir_))[idx]
    return os.path.join(dir_, file)
getLatestTrack()

imagesDir = "../CameraMotionLogger/Server/website/image_uploads/"
web_images_Dir = "../webgrid/image_uploads/"

ultrasoundDir = "../neuralnet/fastNNServer/outputs/"

static_pictures = os.listdir(imagesDir)[1:]
web_static_pictures = os.listdir(web_images_Dir)[1:]

def true_random_image():
    p = random.choice(static_pictures)
    return np.rot90(np.array(Image.open(imagesDir + p)), 3)

def true_random_web_image():
    p = random.choice(web_static_pictures)
    return np.rot90(np.array(Image.open(web_images_Dir + p)), 3)
class TrackedUS:
    def __init__(self, filename):
        self.track = np.array(pickle.load(open(filename, "rb"))).transpose()
        self.time = self.track[3]
        self.startTime = self.time[0]
        self.endTime = self.time[-1]
        
        #load camera pictures for tracking
        pictures = os.listdir(imagesDir)[1:]

        self.p_timestamps = []

        self.p_arrays = []
        for p in pictures:
            t_float = float(p[:-4])
            if self.startTime < t_float < self.endTime:
                try:
                    self.p_timestamps.append(t_float)
                    self.p_arrays.append(np.rot90(np.array(Image.open(imagesDir + p)), 3))
                except OSError:
                    print("bad image", p)
        #load ultrasound images
        
        for us in os.listdir(ultrasoundDir):
            usStart, usEnd = [float(timeStamp) for timeStamp in us.split("_")]
            
            if usStart < self.startTime < self.endTime < usEnd:
                print("yay")
                self.ultrasound = pickle.load(open(ultrasoundDir + us, "rb"))
                break
        else:
            print("no ultrasound found that covers the time of the track")
            
    
    def getAttitudeFromTimestamp(self, t):
        track = self.track
        return [np.interp([t], track[3], track[0]), 
                np.interp([t], track[3], track[1]),
                np.interp([t], track[3], track[2])]
    
    def playUS(self):
        for el in self.ultrasound:
            cv2.imshow("ultra", el[1])
            cv2.waitKey(10)
    def playVideo(self):
        for arr in self.p_arrays:
            cv2.imshow("ultra", arr)
            cv2.waitKey(10)
    def play(self):
        us_index = 0
        video_index = 0
        try:
            while True:
                if self.p_timestamps[video_index] < self.ultrasound[us_index][0]:
                    cv2.imshow("vid", self.p_arrays[video_index])
                    cv2.waitKey(10)
                    video_index += 1
                else:
                    cv2.imshow("ultra", self.ultrasound[us_index][1])
                    cv2.waitKey(10)
                    us_index += 1
        except IndexError:
            print(us_index, video_index)
        print(done)
    def play2(self):
        us_index = 0
        video_index = 0
        try:
            while True:
                
                if self.p_timestamps[video_index] < self.ultrasound[us_index][0]:
                    arr = self.p_arrays[video_index][::3, ::3]
                    v_timestamp = self.p_timestamps[video_index]
                    out = scipy.ndimage.rotate(arr, 
                        -self.getAttitudeFromTimestamp(v_timestamp)[1] * 360 / (2 * np.pi))
                    cv2.imshow("vid", out)
                    cv2.waitKey(10)
                    video_index += 1
                else:
                    arr = self.ultrasound[us_index][1][::3, ::3]
                    u_timestamp = self.ultrasound[us_index][0]
                    out = scipy.ndimage.rotate(arr, 
                        -self.getAttitudeFromTimestamp(u_timestamp)[1] * 360 / (2 * np.pi))
                    cv2.imshow("ultra", out)
                    cv2.waitKey(10)
                    us_index += 1
        except IndexError:
            print(us_index, video_index)
        print("done")


def intersection(line1, line2):
    """Finds the intersection of two lines given in Hesse normal form.

    Returns closest integer pixel locations.
    See https://stackoverflow.com/a/383527/5087436
    """
    rho1, theta1 = line1[0]
    rho2, theta2 = line2[0]
    A = np.array([
        [np.cos(theta1), np.sin(theta1)],
        [np.cos(theta2), np.sin(theta2)]
    ])
    b = np.array([[rho1], [rho2]])
    x0, y0 = np.linalg.solve(A, b)
    #x0, y0 = x0, np.round(y0))
    return [[x0, y0]]

def transform(image, attitude=None, graph=False):
    
    avoid_singularity = False
    for _ in range(2):
        try:
            
            
            
                
            
            def sort_lines(l, line_to_sort_intersections):
                
                
                def intersection_coord(line):
                    inter = intersection(line, line_to_sort_intersections[0])
                    return inter[0][0]
                return np.array(sorted(list(l), key=intersection_coord))


            if attitude is not None:
                roll, yaw, pitch = attitude
                #print(attitude)
                
                
                #compensate for camera misalignment in body
                roll -= .011
                pitch -= .035
            else:
                roll, yaw, pitch = [0],[0],[0]
            
            l2, l1 = getLines.get_lines(image, graph=graph, avoid_singularity=avoid_singularity, liveCV=True)
            cv2.waitKey(1)
            
            if l1 is None:
                return 1000, np.zeros(7)

            if len(l1) == 0 or len(l2) == 0:
                return 1000, np.zeros(7)

            if len(l1) == 1 and len(l2) == 1:
                return 1000, np.zeros(7)

            l1 = sort_lines(l1, l2)
            l2 = sort_lines(l2, l1)
            
            if graph:
                plt.plot(l1[:, 0, 0], l1[:, 0, 1])
                plt.plot(l2[:, 0, 0], l2[:, 0, 1])
                plt.show()



            for _ in range(2):
                #print(l1)
                #print(l2)
                if attitude is not None:
                    activeParams=np.array([True, True, True, False, False, True, False])
                else:
                    activeParams=np.array([True, True, True, True, True, True, False])


                m = auto_align.ManualAligner(image, activeParams=activeParams)

                m.vector = np.array([0, 0, -2.5, -pitch[0] * 180 / np.pi, -roll[0]* 180 / np.pi, -yaw[0]* 180 / np.pi + 25 + 90, (4/3) * 194])
                #print(m.vector)
                #print(m.gridOnScreen.shape)
                pts = []
                for i, line1 in enumerate(l1):
                    for j, line2 in enumerate(l2):
                        x = intersection(line1, line2)[0]
                        pts.append(x)

                        m.gridOnScreen[0, 12 * (5 + i) + (4 + j)] = x[0]
                        m.gridOnScreen[1, 12 * (5 + i) + (4 + j)] = x[1]
                pts = np.array(pts)
                #m.interactive_align()
                #m.align_from_saved(m.gridOnScreen)
                
                m.align_from_saved(m.gridOnScreen)
                if graph:
                    m.present_registration()
                    print(m.error(m.vector))
                    print(m.vector)
                if m.vector[2] <= 0 and m.error(m.vector) < 200:
                    break
                l1, l2 = l2, l1
            
        except ValueError as e:
            print(e)
            return 1000, np.zeros(7)
        if m.error(m.vector) < 200:
            break
        avoid_singularity=True
        
    return m.error(m.vector), m.vector

def registerFromLines(lines, attitude=None, graph=False, vertical_factor = 1, focalLength = 1):
    print("attitude")
    try:
        def sort_lines(l, line_to_sort_intersections):
            def intersection_coord(line):
                inter = intersection(line, line_to_sort_intersections[0])
                return inter[0][0]
            return np.array(sorted(list(l), key=intersection_coord))


        if attitude is not None:
            roll, yaw, pitch = attitude
            #print(attitude)
            
            
            #compensate for camera misalignment in body
            roll -= .011
            pitch -= .035
        else:
            roll, yaw, pitch = [0],[0],[0]
        
        l2, l1 = getLines.split(lines)
        
        if l1 is None:
            return 1000, np.zeros(7)

        if len(l1) == 0 or len(l2) == 0:
            return 1000, np.zeros(7)

        if len(l1) == 1 and len(l2) == 1:
            return 1000, np.zeros(7)

        l1 = sort_lines(l1, l2)
        l2 = sort_lines(l2, l1)
        
        if 0:#graph:
            plt.plot(l1[:, 0, 0], l1[:, 0, 1])
            plt.plot(l2[:, 0, 0], l2[:, 0, 1])
            plt.show()



        for _ in range(2):
            #print(l1)
            #print(l2)
            if attitude is not None:
                activeParams=np.array([True, True, True, False, False, True, False])
            else:
                activeParams=np.array([True, True, True, True, True, True, False])


            m = auto_align.ManualAligner(np.ones((int(128 * vertical_factor), 128)), activeParams=activeParams)

            m.vector = np.array([0, 0, -2.5, -pitch[0] * 180 / np.pi, -roll[0]* 180 / np.pi, -yaw[0]* 180 / np.pi + 25 + 90, (4/3) * 117])
            #print(m.vector)
            #print(m.gridOnScreen.shape)
            pts = []
            for i, line1 in enumerate(l1):
                for j, line2 in enumerate(l2):
                    x = intersection(line1, line2)[0]
                    pts.append(x)

                    m.gridOnScreen[0, 12 * (5 + i) + (4 + j)] = x[0]
                    m.gridOnScreen[1, 12 * (5 + i) + (4 + j)] = x[1] * vertical_factor
            pts = np.array(pts)
            #m.interactive_align()
            #m.align_from_saved(m.gridOnScreen)
            
            m.align_from_saved(m.gridOnScreen)
            print(m.error(m.vector))
            if m.vector[2] <= 0 and m.error(m.vector) < 30:
                
                break
            l1, l2 = l2, l1
        
    except ValueError as e:
        print(e)
        return 1000, np.zeros(7)
    
    if graph:
                    m.present_registration()
                    print(m.error(m.vector))
                    print(m.vector)  
        
    return m.error(m.vector), m.vector

def transform_from_sequence(us_seq, idx, graph):
    return transform(us_seq.p_arrays[idx][::3, ::3], us_seq.getAttitudeFromTimestamp(us_seq.p_timestamps[idx]), graph=graph)