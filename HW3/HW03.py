from PIL import Image
import math
import numpy as np

IMAGE_PATH = 'iguana.bmp'
SIGMA = 1.6
STRONG = 255
WEAK = 80

def arrayToImg(array, name):
    array = array.astype(np.uint8)
    thinEdgeImg = Image.fromarray(array)
    # thinEdgeImg.show()
    thinEdgeImg.save(name)

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

def convolve2d(array, filter):
    # filter 좌우상하 반전
    filter = np.flip(filter)
    # 연산을 위해 float로 설정
    array = array.astype(np.float32) 
    filter = filter.astype(np.float32)
    # filter의 크기에 따라 padding 추가
    paddingSize = int((filter[0].size - 1) / 2)
    padded2d = np.pad(array, ((paddingSize, paddingSize), (paddingSize, paddingSize)))
    # 원본과 동일한 크기의 result 배열 생성 
    rowSize = array.shape[0]
    colSize = array.shape[1]
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

def gaussconvolve2d(array, sigma):
    gaussFilter = gauss2d(sigma)
    return convolve2d(array, gaussFilter)

def sobel_filters(img):
    # Derivate Kernel을 Convolution하여 Ix, Iy 구함
    IxKernel = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    IyKernel = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    Ix = convolve2d(img, IxKernel)
    Iy = convolve2d(img, IyKernel)
    
    G = np.hypot(Ix, Iy) # sqrt(Ix^2 + Iy^2)
    theta = np.arctan2(Iy, Ix) # tan^(-1)(Iy / Ix)

    # mapping (0 to 255)
    maxVal = G.max()
    minVal = G.min()
    G = (G * 255) / (maxVal - minVal)
    return (G, theta)

def getDirection(degree):
    # 가장 가까운 방향을 찾아냄
    directionArray = np.array([0, 45, 90, 135, 180])
    index = (np.abs(directionArray - degree)).argmin()
    return directionArray[index % 4]

def non_max_suppression(G, theta):
	# reduce range : 2 * pi => pi
	theta = (theta + np.pi) % np.pi
	# radian to degree
	degrees = theta * 180 / np.pi

	thinEdge = np.zeros(G.shape)
	ySize, xSize = G.shape
	for y in range(1, ySize - 1): # 이미지 끝 부분은 제외
		for x in range(1, xSize - 1):
			areaG = G[y - 1 : y + 2, x - 1 : x + 2]
			# degree에 따라 확인할 target 변경
			degree = getDirection(degrees[y][x])
			target = np.array([0, 0, 0])
			if(degree == 0):     target = areaG[1, :]
			elif(degree == 45):  target = np.diag(np.fliplr(areaG))
			elif(degree == 90):  target = areaG[:, 1]
			elif(degree == 135): target = np.diag(areaG)
			else: assert True, "direction is not valid"
	
			# pixel이 지정 방향에서 max gradient를 가지는지 확인
			if(target.argmax() == 1):
				thinEdge[y][x] = G[y][x]
	return thinEdge

def double_thresholding(img):
    diff = img.max() - img.min()
    highT = img.min() + diff * 0.15
    lowT = img.min() + diff * 0.03
    thresholded = np.where(img > highT, STRONG, img)
    thresholded = np.where((img > lowT) & (img <= highT), WEAK, thresholded)
    thresholded = np.where(img <= lowT, 0, thresholded)
    return thresholded

def DFS(img, centerY, centerX):
    ySize, xSize = img.shape
    # 이미지 끝 부분은 제외
    if(centerY <= 0 or centerX <= 0 or centerY >= ySize - 1 or centerX >= xSize - 1):
        return
    # 연결된 모든 weak edge를 strong edge로 변환
    for y in range(centerY - 1, centerY + 2):
        for x in range(centerX - 1, centerX + 2):
            if(img[y][x] == WEAK):
                img[y][x] = STRONG
                DFS(img, y, x)

def hysteresis(img):
    # 모든 strong edge를 순회하여 DFS
    strongY, strongX = np.where(img == STRONG)
    for i in range(len(strongY)):
        DFS(img, strongY[i], strongX[i])
    # 나머지 weak edge는 제거
    img = np.where(img == WEAK, 0, img)
    return img

def problem1():
    img = Image.open(IMAGE_PATH)
    imgGrey = img.convert('L')
    arrayGrey = np.asarray(imgGrey)
    arrayGreyBlur = gaussconvolve2d(arrayGrey, SIGMA)
    arrayToImg(arrayGreyBlur, 'problem1.png')
    return arrayGreyBlur

def problem2(): 
    arrayGreyBlur = problem1()
    G, theta = sobel_filters(arrayGreyBlur)
    arrayToImg(G, 'problem2.png')
    return G, theta

def problem3():
    G, theta = problem2()
    thinEdgeArray = non_max_suppression(G, theta)
    arrayToImg(thinEdgeArray, 'problem3.png')
    return thinEdgeArray

def problem4():
    thinEdgeArray = problem3()
    thresholded = double_thresholding(thinEdgeArray)
    arrayToImg(thresholded, 'problem4.png')
    return thresholded

def problem5():
    thresholded = problem4()
    linked = hysteresis(thresholded)
    arrayToImg(linked, 'problem5.png')
    
if(__name__ == "__main__"):
    problem5()