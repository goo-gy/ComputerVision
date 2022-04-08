from PIL import Image
import numpy as np
import math

def imgToRGBArray(imagePath):
    # RGB 채널 분리하여 numpy 배열로 변환
    img = Image.open(imagePath)
    img.save("ok.png")
    imgRGB = img.split()
    arrayRGB = [ np.asarray(channel) for channel in imgRGB ]
    return arrayRGB

def boxfilter(n):
    assert n % 2 == 1, "Dimension must be odd"
    # element의 합은 1이 되도록 normalize
    value = 1 / n**2
    return np.full((n, n), value)

def gauss1d(sigma):
    # sigma * 6을 올림한 값으로 count 설정
    # count가 짝수라면 count에 1을 더해줌
    count = math.ceil(sigma * 6)
    if(count % 2 == 0):
        count += 1
    # 중간값이 0이고 거리가 1멀어질 수록 절대값이 1증가 하도록 배열 생성
    # X^2로 사용하기 때문에 X의 부호는 상관없다
    halfCount = math.floor(count/2)
    X = np.arange(-halfCount, halfCount + 1)
    ndGaussian = np.exp(-X**2/(2*sigma**2))
    # sum으로 나누어 모든 element의 합이 1이 되도록 normalize한다.
    return ndGaussian / ndGaussian.sum()

def gauss2d(sigma):
    # gaussX = e^(-X^2/(2*sigma^2))
    # gaussY = e^(-Y^2/(2*sigma^2))
    # 이므로 외적하면 각 x, y에 대해서 e^(-(X^2 + Y^2)/(2*sigma^2))
    gaussX = gauss1d(sigma)
    gaussY = gauss1d(sigma)
    return np.outer(gaussX, gaussY)

def convolve2d(array2d, filter):
    # 연산을 위해 float로 설정
    array2d = array2d.astype(np.float32) 
    filter = filter.astype(np.float32)
    # filter의 크기에 따라 padding 추가
    paddingSize = int((filter[0].size - 1) / 2)
    padded2d = np.pad(array2d, ((paddingSize, paddingSize), (paddingSize, paddingSize)))
    # 원본과 동일한 크기의 result 배열 생성 
    rowSize = array2d.shape[0]
    colSize = array2d.shape[1]
    result2d = np.zeros((rowSize, colSize))

    # 모든 픽셀에 주변 영역과 filter를 곱한 배열의 sum으로 값 설정
    # filter가 normalize되어 있기 때문에 mean이 아닌 sum을 사용
    for i in range(0, rowSize):
        for j in range(0, colSize):
            range2d = padded2d[i : i + 2*paddingSize + 1,
                               j : j + 2*paddingSize + 1]
            filteredPixel = range2d * filter
            result2d[i][j] = filteredPixel.sum()
    return result2d

def gaussconvolve2d(array2d, sigma):
    gaussFilter = gauss2d(sigma)
    # filter 좌우상하 반전
    filter = np.flip(gaussFilter)
    return convolve2d(array2d, filter)

def imageBlurGrey(imagePath, sigma):
    img = Image.open(imagePath)
    imgGrey = img.convert('L')
    originGrey = np.asarray(imgGrey)
    blurGrey = gaussconvolve2d(originGrey, sigma)
    blurGrey = blurGrey.astype(np.uint8)
    imgGreyBlur = Image.fromarray(blurGrey)
    imgGrey.save('dogGrey.png')
    imgGreyBlur.save('dogGreyBlur.png')
    imgGrey.show()
    imgGreyBlur.show()

def lowPassFilter(imagePath, sigma):
    originRGB = imgToRGBArray(imagePath)
    # gaussian convolution을 통해 low frequency 필터링
    lowRGB = [ gaussconvolve2d(channel, sigma) for channel in originRGB ]
    return lowRGB

def highPassFilter(imagePath, sigma):
    originRGB = imgToRGBArray(imagePath)
    # gaussian convolution을 통해 low frequency 필터링
    lowRGB = [ gaussconvolve2d(channel, sigma) for channel in originRGB ]
    # 원본에서 low frequency 이미지를 빼서 high frequence 필터링
    highRGB = [ originRGB[channelNum] - lowRGB[channelNum] for channelNum in range(3) ]
    return highRGB

def saveImageFromArrayRGB(arrayRGB, saveName):
    # overflow, underflow가 일어나지 않도록 값 조정
    for channel in arrayRGB:
        channel[np.where(channel > 255)] = 255
        channel[np.where(channel < 0)] = 0
    # 이미지로 저장하기 위해 type 변환
    arrayRGB = [ channel.astype(np.uint8) for channel in arrayRGB ]
    imgDetail3 = [ Image.fromarray(channel) for channel in arrayRGB ]
    imgDetail = Image.merge("RGB", imgDetail3)
    imgDetail.show()
    imgDetail.save(saveName)

def makeBlurImage(imagePath, sigma, saveName):
    lowRGB = lowPassFilter(imagePath, sigma)
    saveImageFromArrayRGB(lowRGB, saveName)

def makeDetailImage(imagePath, sigma, saveName):
    highRGB = highPassFilter(imagePath, sigma)
    # high pass filter는 image - image 수행하여 값이 하향 평준화 되어 있음
    # 128을 더해 값 상향
    highRGB = [ channel + 128 for channel in highRGB ]
    saveImageFromArrayRGB(highRGB, saveName)

def makeHybridImage(imagePath1, imagePath2, sigma, saveName):
    # low frequency + high frequency
    lowRGB = lowPassFilter(imagePath1, sigma)
    highRGB = highPassFilter(imagePath2, sigma)
    hybridRGB = [ lowRGB[channelNum] + highRGB[channelNum] for channelNum in range(3) ]
    saveImageFromArrayRGB(hybridRGB, saveName)

# print("boxfilter(3)\n", boxfilter(3))
# print("boxfilter(7)\n", boxfilter(7))
# print("boxfilter(4)")
# print(boxfilter(4))

# print("gauss1d(0.3)\n", gauss1d(0.3)) # 0.3, 0.5, 1, 2
# print("gauss1d(0.5)\n", gauss1d(0.5))
# print("gauss1d(1)\n", gauss1d(1))
# print("gauss1d(2)\n", gauss1d(2))

# print("gauss2d(0.5)\n", gauss2d(0.5))
# print("gauss2d(1)\n", gauss2d(1))

# result = gaussconvolve2d(boxfilter(5), 0.5)
# imageBlurGrey("hw2_image/2b_dog.bmp", 3)
# makeBlurImage("hw2_image/2b_dog.bmp", 5, "lowFrequencyDog.png")
# makeDetailImage("hw2_image/2a_cat.bmp", 5, "highFrequencyCat.png")
makeHybridImage("hw2_image/2b_dog.bmp", "hw2_image/2a_cat.bmp", 5, "hybridDogCat.png")
