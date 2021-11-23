import cv2
from math import floor
import matplotlib.pyplot as plt
import numpy as np


class Halftoning:
    def __init__(self, image, method = 'fs'):
        self.image = image.copy()
        if method == 'fs':
            self.method = self.floyd_steinberg
        elif method == 'dt':
            self.method = self.noise
        else:
            print('Not a known method')
        self.output = image.copy()

    def normalized(self, pix, min=0, max=255):
        f = ((pix - min) / (max - min)) - 0.5
        return f

    def threshold(self, pix, thresh=127):
        return 255 * floor(pix / (thresh))

    def floyd_steinberg(self, resize=None, min=0, max=255):
        if isinstance(resize, tuple):
            output = cv2.resize(self.output, resize)
        else:
            output = self.output.copy()
        
        w, h = output.shape
        
        for j in range(0, h):
            for i in range(0, w):
                fq = self.threshold(output[i][j])
                err = output[i][j] - fq * (max - min) - min 
                output[i][j] = fq

                if(i < w-1):
                    output[i+1][j  ] += round(err * 7/16)
                if(i > 1 and j < h-1):
                    output[i-1][j+1] += round(err * 3/16)
                if(j < h-1):
                    output[i  ][j+1] += round(err * 5/16)
                if(i < w-1 and j < h-1):
                    output[i+1][j+1] += round(err * 1/16)

        self.output = output

    def noise(self, resize=None, min=0, max=255):
        if isinstance(resize, tuple):
            output = cv2.resize(self.output, resize)
        else:
            output = self.output.copy()
        
        w, h= output.shape

        noise = np.random.uniform(-np.pi, np.pi, size=(w, h))
        plt.imshow(noise)

        for x in range(w):
            for y in range(h):
                if (x-w/2)**2+(y-h/2)**2 < 10000:
                    noise[x][y] /= 3.14
        plt.figure()
        plt.imshow(noise)

        idft_noise = np.fft.ifft2(noise)
        mask = idft_noise.real # - min / (max - min)
        output = output - min / (max - min)

        plt.figure()
        plt.imshow(np.absolute(idft_noise))

        plt.show()
        output = cv2.add(output, mask, dtype=cv2.CV_8U)
        _, output = cv2.threshold(output, 127, 255, cv2.THRESH_BINARY)

        self.output = output.copy()
    
    def run(self, resize=None, min=0, max=1):
        if isinstance(resize, tuple):
            output = cv2.resize(self.image, resize)
        else:
            output = self.image.copy()
        
        cv2.imshow('Input', output)
        return self.method(resize, min, max)

    def show(self):

        cv2.imshow('Output', self.output)

    def save(self, filename='output'):
        cv2.imwrite(filename, self.output)

def color_fs(image, resize = None):
        b, g, r = cv2.split(image)
        
        fs_red = Halftoning(r, 'fs')
        fs_red.run(resize)
        fs_green = Halftoning(g, 'fs')
        fs_green.run(resize)
        fs_blue = Halftoning(b, 'fs')
        fs_blue.run(resize)
    
        return cv2.merge([fs_blue.output, fs_green.output, fs_red.output]) 
