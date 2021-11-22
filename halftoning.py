import numpy as np
import cv2
from PIL import Image
from math import floor

class Halftoning:
    def __init__(self, image, method = 'fs'):
        self.image = image.copy()
        if method == 'fs':
            self.method = self.floyd_steinberg
        else:
            self.method = self.dither
        self.output = image.copy()

    def normalized(self, pix, min=0, max=255):
        f = ((pix - min) / (max - min)) - 0.5
        return f

    def threshold(self, pix, thresh=128):
        return 255 * floor(pix / thresh)
        # if pix>=thresh:
        #     return 1
        # return 0

    def floyd_steinberg(self, resize=None, min=0, max=255):
        if isinstance(resize, tuple):
            output = cv2.resize(self.output, resize)
        else:
            output = self.output.copy()
        
        w, h = output.shape
        
        for j in range(1, h):
            for i in range(1, w):
                # fq = self.threshold(self.normalized(output[i][j]))
                # err = output[i][j] - fq*(max - min) - min 

                pix = output[i][j]
                new_pix = self.threshold(pix)
                output[i][j] = new_pix
                err = pix - new_pix

                if(i < w-1):
                    output[i+1][j  ] += round(err * 7/16)
                if(i > 1 and j < h-1):
                    output[i-1][j+1] += round(err * 3/16)
                if(j < h-1):
                    output[i  ][j+1] += round(err * 5/16)
                if(i < w-1 and j < h-1):
                    output[i+1][j+1] += round(err * 1/16)

                # output[i][j] = self.threshold(output[i][j], 127) * (max-min) + min
        self.output = output.copy()

    def dither(self, min=0, max=255):
        w, h= self.image.shape

        noise = np.random.random(size=(w, h))
        output = self.normalized(self.image)
        output = cv2.add(output, noise)

        for j in range(h-1):
            for i in range(1, w-1):
                output[i][j] = self.threshold(output[i][j], 1)

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

def color_fs(image):
        b, g, r = cv2.split(image)
        
        fs_red = Halftoning(r, 'fs')
        fs_red.run((128, 128))
        fs_green = Halftoning(g, 'fs')
        fs_green.run((128, 128))
        fs_blue = Halftoning(b, 'fs')
        fs_blue.run((128, 128))
    
        return cv2.merge([fs_blue.output, fs_green.output, fs_red.output]) 

def main():
    filename = 'abe.jpg'
    image = cv2.imread(filename)
    image_bw = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    floyd_steinberg = Halftoning(image_bw, 'fs')
    floyd_steinberg.run()
    floyd_steinberg.show()
    floyd_steinberg.save('fs'+filename)

    # output = color_fs(image)
    # cv2.imshow('Output', output)
    # cv2.imwrite('fs_color_'+filename, output)

    while(cv2.waitKey(0) != 27):
        continue

    cv2.destroyAllWindows()
    pass

if __name__=='__main__':
    main()