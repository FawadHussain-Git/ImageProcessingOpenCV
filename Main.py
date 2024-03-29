from OpenCV import ImageProcess
import cv2

def main():

    # imageProcess = ImageProcess("AM04NES.JPG")
    while(1):
        cap = cv2.VideoCapture(0)
        _, frame = cap.read()
        cv2.imshow('frame', frame)
        print(type(frame))
        k = cv2.waitKey(5) & 0xFF
        if k == 27:
            break
    cv2.destroyAllWindows()
    cap.release()
    
    imageProcess = ImageProcess(frame)
    # print("Image before resize")
    # h, w = imageProcess.getDiminsion()
    #
    # print(h)
    # print(w)
    #
    # imageProcess.getRotatedImage(180)
    # rotatedImage = imageProcess.getImage()
    # # imageProcess.showImage()


    # imageProcess.resizeImage(150,150)
    # resizedLImage = imageProcess.getImage()

    # print("After Resize")
    # h,w = imageProcess.getDiminsion()
    #
    # print(h)
    # print(w)
    # imageProcess.writeImage("resizedImage.png")
    #
    # imageProcess.resizeImage(h//2,w//2)
    # imageProcess.writeImage("resizedSmall.png")
    # resizedSImage = imageProcess.getImage()
    #
    # imageProcess.invertImage()
    # imageProcess.writeImage("invertedImage.png")

    # imageProcess.commonRegion("resizeImage.png")
    # imageProcess.writeImage("commonRegion.png")

    # imageProcess.showImageList()
    # imageProcess.showImage()

    # imageProcess.meanBlurFilter(9)
    # imageProcess.imageComparsionRGB()
    # imageProcess.imageComparsionGray()

    # imageProcess.gaussianBlurFilter(9)
    # imageProcess.imageComparsionRGB()
    # imageProcess.imageComparsionGray()

    # imageProcess.medianBlurFilter(9)
    # imageProcess.imageComparsionRGB()
    # imageProcess.imageComparsionGray()

    # imageProcess.laplacianFilterEdgeDetection(9)
    # imageProcess.imageComparisionOf3Images()

    # imageProcess.frequencyFilter()
    # imageProcess.imageComparsionGray()

    # imageProcess.freqToSpatial()

    # imageProcess.unsharpFilter()
    # imageProcess.imageComparsionGray()

    # imageProcess.conservative_smoothing_gray(5)
    # imageProcess.imageComparsionGray()


if __name__ == "__main__":
    main()
