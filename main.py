import numpy as np
import cv2

from halftoning import Halftoning

def main():
    filename = 'Gray.png'
    image_bw = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)

    dithering = Halftoning(image_bw, 'fs')
    dithering.run((512, 512))
    # dithering.save('fs_Gradient.jpg')
    dithering.show()

    while(cv2.waitKey(0) != 27):
        continue
    cv2.destroyAllWindows()

if __name__=='__main__':
    main()