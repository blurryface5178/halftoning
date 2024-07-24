import cv2
import numpy as np
image = cv2.imread("./test1.jpg",cv2.IMREAD_GRAYSCALE)

image = image.astype(float)

class HalftoneRemoval:
    def __init__(self,image:np.ndarray):
        """
        image : cv2 Grayscale Image
        """
        self.__image = image.astype(float)
        self.output = None
        self.__dehalftoning()

    def __generate_grid(self,m, n):
        x = np.arange(-m/2, m/2) / m
        y = np.arange(-n/2, n/2) / n
        z = np.zeros((m, n))
        mask = np.zeros((m, n))
        for i in range(m-1):
            for j in range(n-1):
                z[i][j] = x[i] ** 2 + y[j] ** 2
        return z

    def __generate_mask(self,m, n, a=1.0, b=.19):
            z = self.__generate_grid(m, n)
            mask = a * np.exp(-np.pi*z / b**2)
            return mask

    def __dehalftoning(self):
        w, h = self.__image.shape

        h, w = self.__image.shape
        
        F = np.fft.fft2(image)
        F = np.fft.fftshift(F)

        mask = self.__generate_mask(w, h)

        product = np.multiply(F.T, mask)
        
        ifft_product = np.fft.ifft2(product)
        ifft_mag = np.abs(ifft_product) ** 2

        normalized = ifft_mag.T / ifft_mag.max() * 255

        self.output = normalized.astype(int)

    def save(self,save_path="./out",format="png"):
        if format in ['jpg','png','webp','gif']:
            cv2.imwrite(f"{save_path}.{format}",self.output)
        else:
            Exception("올바른 이미지 포맷을 적어주세요.")

if __name__ == "__main__":
    image = cv2.imread("./test1.jpg",cv2.IMREAD_GRAYSCALE)
    hr = HalftoneRemoval(image)
    hr.save()