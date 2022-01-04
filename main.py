import numpy as np
import cv2

from halftoning import Halftoning

def main():
    filename = "image.jpg"
    image_bw = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)

    fs = Halftoning(image_bw)
    fs.run()
    fs.show()
    fs.save('fs'+filename)

    while(cv2.waitKey(0) != 27):
        continue
    cv2.destroyAllWindows()

if __name__=='__main__':
    main()