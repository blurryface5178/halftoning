import cv2
from math import floor
import matplotlib.pyplot as plt
import numpy as np

class Halftoning:
    def __init__(self, image, method = 'fs'):
        self.image = image.astype(float)
        if method == 'fs':
            self.method = self.floyd_steinberg
        elif method == 'blue':
            self.method = self.blue_noise
        elif method == 'remove':
            self.method = self.dehalftoning
        else:
            print('Not a known method')
        self.output = self.image.copy()
    
    def threshold(self, pix, thresh=255):
        return 255 * floor(pix / thresh)
    
    def normalized(self, array):
        return (array.real - array.min()) / (array.max() - array.min())

    def floyd_steinberg(self, resize=None):
        output = self.output.copy()

        if isinstance(resize, tuple):
            output = cv2.resize(output, resize)

        w, h = output.shape

        for j in range(0, h):
            for i in range(0, w):
                fq = self.threshold(output[i][j], 255)
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
    
    def generate_grid(self, m, n):
        # This portion of code is supposed to speed by the computation by 4x but its a little buggy
        # Hence we end up withh the slow functional code
        # I'm working on another method that should be even faster

        # z = np.zeros((m, n)).astype(float)
        # for i in range(int(m/2)-1):
        #     for j in range(int(n/2)-1):
        #         z[int(m/2)+i][int(n/2)+j] = (float(i)/m) ** 2 + (float(j)/n) ** 2
    
        # z[ : int(m/2),int(n/2):] = np.flip(z[int(m/2): ,int(n/2): ], 0)
        # z[ : , :int(n/2)] = np.flip(z[:, int(n/2):n], 1)
        
        x = np.arange(-m/2, m/2) / m
        y = np.arange(-n/2, n/2) / n
        z = np.zeros((m, n))
        mask = np.zeros((m, n))
        for i in range(m-1):
            for j in range(n-1):
                z[i][j] = x[i] ** 2 + y[j] ** 2
        return z

    def generate_noise(self, m, n, a=1, b=100):
        phase = np.random.normal(size=(m, n)) * np.pi
        z = self.generate_grid(m, n)
        mag = 1 - np.exp(-np.pi*z / b**2)
        noise = mag * (np.cos(phase) + 1j * np.sin(phase))

        print('mag', mag*255)
        cv2.imshow('mag', z)
        cv2.imshow('phase', phase)

        inverse_noise = np.fft.ifft2(noise)
        in_real = self.normalized(inverse_noise)
        return in_real

    def blue_noise(self, resize=None):
        if isinstance(resize, tuple):
            output = cv2.resize(self.output, resize)
        else:
            output = self.output.copy()
        
        w, h= output.shape

        noise = self.generate_noise(w, h)
        show = np.abs(noise) * 255.0
        cv2.imwrite('bluenoise.png', show.astype(int))

        output = self.normalized(output)

        added = cv2.add(output, noise.real)
        _, added = cv2.threshold(added, 0.9, 2, cv2.THRESH_BINARY)

        self.output = added.copy() * 255
    
    def run(self, resize=None):
        if isinstance(resize, tuple):
            show = cv2.resize(self.image, resize)
        else:
            show = self.image.copy()
        
        # cv2.imshow('Input', show)
        return self.method(resize)

    def show(self):
        cv2.imshow('Output', self.output.astype(np.uint8))

    def save(self, filename='output.jpg'):
        cv2.imwrite(filename, self.output)

    def generate_mask(self, m, n, a=1.0, b=0.03):
        z = self.generate_grid(m, n)
        mask = a * np.exp(-np.pi*z / b**2)
        return mask

    def dehalftoning(self, resize=None):
        w, h = self.image.shape
        
        if w % 2 > 0:
            image_bw = cv2.resize(self.image, (h, w-1))
        elif h % 2 > 0:
            image_bw = cv2.resize(self.image, (h-1, w))

        h, w = self.image.shape
        
        F = np.fft.fft2(self.image)
        F = np.fft.fftshift(F)

        mask = self.generate_mask(w, h)

        product = np.multiply(F.T, mask)
        
        ifft_product = np.fft.ifft2(product)
        ifft_mag = np.abs(ifft_product) ** 2

        normalized = ifft_mag.T / ifft_mag.max() * 255

        self.output = normalized.astype(int)
