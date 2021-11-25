import numpy as np
import cv2

from Halftoning import Halftoning

def main():
    filename = 'cat_hw.jpg'
    image_bw = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)

    dithering = Halftoning(image_bw, 'fs')
    dithering.run()
    dithering.save('cat_hw_out.jpg')
    dithering.show()

    while(cv2.waitKey(0) != 27):
        continue
    cv2.destroyAllWindows()

if __name__=='__main__':
    main()