import numpy as np
import cv2

from halftoning import Halftoning

def main():
    filename = 'Gradient.png'
    image_bw = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)

    dithering = Halftoning(image_bw, 'dt')
    dithering.run((256, 256))
    dithering.save('gradient_dt.jpg')
    dithering.show()

    while(cv2.waitKey(0) != 27):
        continue
    cv2.destroyAllWindows()

if __name__=='__main__':
    main()