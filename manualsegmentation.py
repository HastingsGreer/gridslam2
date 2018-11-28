
import numpy as np
from PIL import Image
import os
import matplotlib.pyplot as plt

import cv2

folder = "test_set/iphone_video_frames/"
outfolder = "test_set/manual_segmentations/"

class ManualSegmenter:
    def __init__(self, image):

        self.image = image

        self.mask = image * 0

        self.line_idx = 0

        self.line = [(0, 0), (0, 0)]
        self.interactive_align()


    def interactive_align(self):
        self.fig = plt.figure()

        self.ax = self.fig.add_subplot(111)
        self.im_disp_obj = self.ax.imshow(self.image)
        self.mask_disp_obj = self.ax.imshow(self.mask)
        

        self.ax.set_title('click on points')
        
        self.fig.canvas.mpl_connect("button_press_event", self.on_click)
        self.fig.canvas.mpl_connect("key_press_event", self.on_key_press)
       
        plt.show()
   

    def draw(self):
        self.im_disp_obj.set_data(self.image)
        self.mask_disp_obj.set_data(self.mask)
        self.fig.canvas.draw()
    def on_click(self, event):
        print(event)
        self.line[self.line_idx] = int(event.xdata), int(event.ydata)
        if self.line_idx == 0:
            
            self.line_idx = 1;
        else:
            self.oldmask = self.mask.copy()
            cv2.line(self.mask,self.line[0],self.line[1],(255,0,0, 255),1)
            self.draw()
            
            self.line_idx = 0;

       
        

    def on_key_press(self, event):
        if event.key == 'a':
            self.image = self.image * 2
            self.draw()
            

        

        if event.key == 'u':
            self.mask = self.oldmask
            self.draw()
            self.line_idx = 0
        
        if event.key == 'd':
            self.gridOnScreen[:, self.activeIndex] = 0, 0
            self.process_updated_annotation()
import random
already_segmented = set(os.listdir(outfolder))
candidates = os.listdir(folder)
random.shuffle(candidates)
for fname in candidates:
    if fname not in already_segmented:
        full_name = os.path.join(folder, fname)
        print(full_name)

        if full_name[-4:] in ['.jpg', '.png']:
            t2 = np.array(Image.open(full_name))
            
            m = ManualSegmenter(t2)
            t2[:, :, 3] = 255 - m.mask[:, :, 3]
            Image.fromarray(t2).save(outfolder + fname)

                