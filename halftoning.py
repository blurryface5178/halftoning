import cv2
from math import floor
import matplotlib.pyplot as plt
import numpy as np


class Halftoning:
    def __init__(self, image, method = 'fs'):
        self.image = image.astype(float)
        if method == 'fs':
            self.method = self.floyd_steinberg
        elif method == 'dt':
            self.method = self.dithering
        else:
            print('Not a known method')
        self.output = self.image.copy()

    def normalized(self, image, min=0, max=255):
        w, h = image.shape
        for y in range(h):
            for x in range(w):
                image[x][y] = ((image[x][y] - min) / (max - min)) - 0.5
        return image
    
    def threshold(self, pix, thresh=0):
        return 255 * floor(pix / thresh)

    def floyd_steinberg(self, resize=None, min=0, max=255):
        # output = self.normalized(self.output)
        output = self.output.copy()

        if isinstance(resize, tuple):
            output = cv2.resize(output, resize)

        w, h = output.shape
        err = 0

        for j in range(0, h):
            for i in range(0, w):
                fq = self.threshold(output[i][j], 127)
                err = output[i][j] - fq
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

    def generate_noise(self, w, h, b=9):
        phase = np.random.normal(size=(w, h)) * np.pi
        x = np.arange(-w/2, w/2)
        y = np.arange(-h/2, h/2)
        z = np.zeros((w, h))
        for i in range(w-1):
            for j in range(h-1):
                z[i][j] = x[i] ** 2 + y[j] ** 2
        mag = 1 - np.exp(-np.pi*z / b**2)
        noise = mag * (np.cos(phase) + 1j * np.sin(phase))

        inverse_noise = np.fft.ifft2(noise)
        in_real = (inverse_noise.real - inverse_noise.real.min()) / (inverse_noise.real.max() - inverse_noise.real.min())
        return in_real

    def dithering(self, resize=None, min=0, max=255):
        if isinstance(resize, tuple):
            output = cv2.resize(self.output, resize)
        else:
            output = self.output.copy()
        
        w, h= output.shape

        noise = self.generate_noise(w, h)
        output = (output - output.min()) / (output.max() - output.min())

        added = cv2.add(output, noise)
        _, added = cv2.threshold(added, 1, 2, cv2.THRESH_BINARY)

        self.output = added.copy() * 255
    
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
