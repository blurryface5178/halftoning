import numpy as np
import cv2

from halftoning import Halftoning

def main():
    filename = "cat_hw.jpg"
    image_bw = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
    print('Working on...'+filename)

    fs = Halftoning(image_bw, 'fs')
    fs.run()
    print('Floyd Steinberg Dithering Complete')

    fs.show()
    fs.save('fs'+filename)

    print('Done...'+filename)

    while(cv2.waitKey(0) != 27):
        continue
    cv2.destroyAllWindows()

if __name__=='__main__':
    main()