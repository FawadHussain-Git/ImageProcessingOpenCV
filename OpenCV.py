import cv2
import os
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageFilter


class ImageProcess:

    def __init__(self,image):

        self.image = image
        self.image2 = None
        self.resultImage = None
        self.resultImage2 = None
        self.filterName = ''


    def colorConversion(self):
        self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2HSV)  # convert to HSV
        self.image2 = cv2.cvtColor(self.image, cv2.COLOR_HSV2BGR)
        self.image2 = cv2.cvtColor(self.image2, cv2.COLOR_BGR2GRAY)


    def readImage(self,image):
        self.image = cv2.imread(image)


    def getImage(self):
        return self.image

    def showImage(self):
        cv2.imshow("image",self.image)
        k = cv2.waitKey(0) & 0xFF

        # wait for ESC key to exit
        if k == 27:
            cv2.destroyAllWindows()

    # def readImage(self,image):
    #     self.image2 =  cv2.imread(image)

    def getDiminsionChannel(self):
        h, w, c =  self.image.shape
        return h , w , c

    def getDiminsion(self):
        h, w, _ = self.image.shape
        return h,w

    def resizeImage(self,height,width):
        self.image = cv2.resize(self.image,(height,width))

    def getRotatedImage(self,degree):
        cols,rows = self.getDiminsion()
        mask = cv2.getRotationMatrix2D((cols / 2, rows / 2), degree, 1)
        self.image = cv2.warpAffine(self.image, mask, (cols, rows))

    def getTranslatedImage(self,mask):
        cols,rows = self.getDiminsion()
        self.image = cv2.warpAffine(self.image, mask, (cols, rows))


    def writeImage(self,filename):
        cv2.imwrite(filename, self.image)


    def invertImage(self):
        self.image = cv2.bitwise_not(self.image)

    # def commonRegion(self):
    #     # print(self.image2.shape)
    #     self.image = cv2.bitwise_and(self.image,self.image2)


    def showImageList(self):
        directory = '/Users/fawadhussain/PycharmProjects/OpenCV'
        os.chdir(directory)
        print(os.listdir(directory))


    def meanBlurFilter(self,filterSize):
        self.filterName = "Mean Filter"
        self.colorConversion()
        figure_size = filterSize  # the dimension of the x and y axis of the kernal.
        self.resultImage = cv2.blur(self.image, (figure_size, figure_size))
        self.resultImage2 = cv2.blur(self.image2,(figure_size, figure_size))

    def gaussianBlurFilter(self,filterSize,sigmax = 0,sigmay=0):
        self.filterName = "Gaussian filter"
        self.colorConversion()
        figure_size = filterSize  # the dimension of the x and y axis of the kernal.
        self.resultImage = cv2.GaussianBlur(self.image, (figure_size, figure_size), 0)
        self.resultImage2 = cv2.GaussianBlur(self.image2, (figure_size, figure_size), 0)


    def medianBlurFilter(self,filterSize):
        self.filterName = "Median Filter"
        self.colorConversion()
        figure_size = filterSize  # the dimension of the x and y axis of the kernal.
        self.resultImage = cv2.medianBlur(self.image, figure_size)
        self.resultImage2 = cv2.medianBlur(self.image2, figure_size)


    def laplacianFilterEdgeDetection(self,filterSize):
        self.filterName = "Laplacian Filter"
        self.colorConversion()
        self.resultImage = cv2.Laplacian(self.image2, cv2.CV_64F)


    def conservative_smoothing_gray(self,filter_size):
        self.filterName = "Conservative Smoothing"
        temp = []
        indexer = filter_size // 2
        self.image2 = self.image
        nrow, ncol = self.getDiminsion()
        for i in range(nrow):
            for j in range(ncol):
                for k in range(i - indexer, i + indexer + 1):
                    for m in range(j - indexer, j + indexer + 1):
                        if (k > -1) and (k < nrow):
                            if (m > -1) and (m < ncol):
                                temp.append(self.image[k, m])
                temp.remove(self.image[i, j])

                max_value = max(temp)
                min_value = min(temp)

                if self.image[i, j] > max_value:
                    self.image2[i, j] = max_value
                elif self.image[i, j] < min_value:
                    self.image2[i, j] = min_value
                temp = []
        return self.image2.copy()


    def frequencyFilter(self):
        self.colorConversion()
        dft = cv2.dft(np.float32(self.image2), flags=cv2.DFT_COMPLEX_OUTPUT)
        # shift the zero-frequncy component to the center of the spectrum
        dft_shift = np.fft.fftshift(dft)
        # save image of the image in the fourier domain.
        self.resultImage2 = 20 * np.log(cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1]))

    def freqToSpatial(self):
        self.colorConversion()
        dft = cv2.dft(np.float32(self.image2), flags=cv2.DFT_COMPLEX_OUTPUT)
        # shift the zero-frequncy component to the center of the spectrum
        dft_shift = np.fft.fftshift(dft)
        rows, cols = self.image2.shape
        crow, ccol = rows // 2, cols // 2
        # create a mask first, center square is 1, remaining all zeros
        mask = np.zeros((rows, cols, 2), np.uint8)
        mask[crow - 30:crow + 30, ccol - 30:ccol + 30] = 1
        # apply mask and inverse DFT
        fshift = dft_shift * mask
        f_ishift = np.fft.ifftshift(fshift)
        img_back = cv2.idft(f_ishift)
        img_back = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])
        # plot both images
        plt.figure(figsize=(11, 6))
        plt.subplot(121), plt.imshow(self.image2, cmap='gray')
        plt.title('Input Image'), plt.xticks([]), plt.yticks([])
        plt.subplot(122), plt.imshow(img_back, cmap='gray')
        plt.title('Low Pass Filter'), plt.xticks([]), plt.yticks([])
        plt.show()

    # def unsharpFilter(self):
    #
    #     image = Image.fromarray(self.image2.astype('uint8'))
    #     self.resultImage2 = image.filter(ImageFilter.UnsharpMask(radius=2, percent=150))


    def imageComparsionRGB(self):
        plt.figure(figsize=(11, 6))
        plt.subplot(121), plt.imshow(cv2.cvtColor(self.image, cv2.COLOR_HSV2RGB)), plt.title('Original')
        plt.xticks([]), plt.yticks([])
        plt.subplot(122), plt.imshow(cv2.cvtColor(self.resultImage, cv2.COLOR_HSV2RGB)), plt.title(self.filterName)
        plt.xticks([]), plt.yticks([])
        plt.show()

    def imageComparsionGray(self):
        plt.figure(figsize=(11, 6))
        plt.subplot(121), plt.imshow(self.image2, cmap='gray'), plt.title('Original')
        plt.xticks([]), plt.yticks([])
        plt.subplot(122), plt.imshow(self.resultImage2, cmap='gray'), plt.title(self.filterName)
        plt.xticks([]), plt.yticks([])
        plt.show()

    def imageComparisionOf3Images(self):
        plt.figure(figsize=(11, 6))
        plt.subplot(131), plt.imshow(self.image2, cmap='gray'), plt.title('Original')
        plt.xticks([]), plt.yticks([])
        plt.subplot(132), plt.imshow(self.resultImage, cmap='gray'), plt.title(self.filterName)
        plt.xticks([]), plt.yticks([])
        plt.subplot(133), plt.imshow(self.image2 + self.resultImage, cmap='gray'), plt.title('Resulting image')
        plt.xticks([]), plt.yticks([])
        plt.show()
