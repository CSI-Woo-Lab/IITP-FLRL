import cv2
import os
import random


def CheckDirectory (path):
	try:
		if not os.path.exists (path):
			os.makedirs(path)
	except OSError:
		print('Error: Failed to create the directory')
		exit(1)


def Video2Frame (source, iMainPath, aMainPath, 
				fileNamePath, bound, cropCnt, capN_max, frameSkip, mode):

	iPath = os.path.join (source, 'video.mov')
	aPath = os.path.join (source, 'annotations.txt')

	if os.path.isfile(iPath) == False:
		print('Error: No Image file, ', iPath)
		exit (1)
	if os.path.isfile(aPath) == False:
		print('Error: No Annotation file, ', aPath)
		exit (1)
	
	cap = cv2.VideoCapture(iPath)

	capW = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
	capH = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
	b_half = int(bound / 2)

	x_center = random.randrange (b_half, capW-b_half+1 - int(capW/2))
	y_center = random.randrange (b_half, capH-b_half+1)
	x_min = x_center - b_half
	x_max = x_center + b_half
	y_min = y_center - b_half
	y_max = y_center + b_half
	
	# cropping images start
	f = open (fileNamePath, 'a')
	capCnt = 0
	frameCnt = 0
	while (cap.isOpened()):
		ret, image = cap.read ()

		if capCnt == capN_max or ret == False:
			break

		if (frameCnt % frameSkip == 0):
			image = image[y_min:y_max, x_min:x_max]
			imageCnt = str(capCnt)
			fileName = iMainPath + '/' + str(cropCnt).zfill(3) + '_' + imageCnt.zfill(6) + '.png'
			f.write (fileName + '\n')
			cv2.imwrite(fileName, image)
			capCnt += 1

		frameCnt += 1
	f.close ()
	cap.release ()

	if capCnt < capN_max:
		print ('Warning !!: capCnt low 2000')

	# cropping labels start
	# origin annotation file format
	# 0. Track ID  1. xmin  2. ymin
	# 3. xmax  4. ymax  5. frame
	# 6. lost  7. occluded  8. generated
	# 9. label

	# origin label
	# 0. Biker  1. Pedestrian  2. Skateboarder
	# 3. Cart  4. Car  5. Bus
	base = list ()

	if mode == 0:
		label_candi = ['Biker']
	elif mode == 1:
		label_candi = ['Biker', 'Pedestrian']

	f = open (aPath, 'r')
	lines = f.readlines ()
	for line in lines:
		line = line.replace('\n', "")
		line = line.replace('\"', "")
		base.append(line.split(' '))
	f.close ()

	for k in range(0, capCnt):
		f = open (aMainPath + '/' + str(cropCnt) +
				'_' + str(k).zfill(6) + '.txt', 'w')
		f.close ()

	for k in range(0,len(base)):
		label = base[k][9]
		frameOr = int(base[k][5])
		imageOr = int(frameOr / frameSkip)
		bxMin = int(base[k][1])
		bxMax = int(base[k][3])
		byMin = int(base[k][2])
		byMax = int(base[k][4])
		bxCen = int( (bxMin + bxMax) / 2 )
		byCen = int( (byMin + byMax) / 2 )
		if (imageOr < capN_max and frameOr % frameSkip == 0 and label in label_candi):
			if (bxCen <= x_max and bxCen >= x_min and byCen <= y_max and byCen >= y_min):
				if bxMin < x_min:
					bxMin = x_min
				if bxMax > x_max:
					bxMax = x_max
				if byMin < y_min:
					byMin = y_min
				if byMax > y_max:
					byMax = y_max
				bxCen = (int( (bxMax + bxMin) / 2) - x_min) / bound
				byCen = (int( (byMax + byMin) / 2) - y_min) / bound
				bxLen = (bxMax - bxMin) / bound
				byLen = (byMax - byMin) / bound
				
				for kk in range (0, len(label_candi)):
					if label == label_candi[kk]:
						label = kk

				f = open (aMainPath + '/' + str(cropCnt) +
							'_' + str(imageOr).zfill(6) + '.txt', 'a')
				data = str(label) + ' ' + str(bxCen) + ' ' + str(byCen) + ' ' + str(bxLen) + ' ' + str(byLen) + '\n'
				f.write (data)
				f.close ()

#########################################################
	# Main Start
#########################################################


# origin folder
# 0. bookstore	1. coupa	2. deathCircle
# 3. gates	4. hyang	5. little
# 6. nexus	7. quad

mode = ['single-detection', 'multi-detection', 'tracking']
scen = ['bookstore', 'coupa', 'deathCircle', 'gates', 'hyang', 'little', 'nexus', 'quad']
videoN = [7, 4, 5, 9, 15, 4, 12, 4]

# settings
cropN = 104
valN = 26
testN = 7
cropBound = 640 # square crop, x-y length
modeOr = 1
scenOr = 3
videoOr = 3
capN_max = 1824
frameSkip = 5

# start
print("Crop Start,", "CropN:", cropN, "CropBound:", cropBound)

mainPath = '/usr/src/app/docker-repository/dataset/crop'
CheckDirectory (mainPath)
mainPath = os.path.join (mainPath, mode[modeOr])
CheckDirectory (mainPath)
mainPath = os.path.join (mainPath, scen[scenOr])
CheckDirectory (mainPath)
iMainPath = os.path.join (mainPath, 'images')
CheckDirectory (iMainPath)
aMainPath = os.path.join (mainPath, 'labels')
CheckDirectory (aMainPath)

trainFile = os.path.join (mainPath, 'train.txt')
valFile = os.path.join (mainPath, 'val.txt')
testFile = os.path.join (mainPath, 'test.txt')

f = open (trainFile, 'w')
f.close ()
f = open (valFile, 'w')
f.close ()
f = open (testFile, 'w')
f.close ()

oPath = 'origin/' + scen[scenOr] + '/video' + str(videoOr)
print("Crop Target: ", oPath)

# crop train
iPath = os.path.join (iMainPath, 'train')
CheckDirectory (iPath)
aPath = os.path.join (aMainPath, 'train')
CheckDirectory (aPath)
for k in range(0, cropN):
	Video2Frame (oPath, iPath, aPath, trainFile, cropBound, k, capN_max, frameSkip, modeOr)
	print('train crop done, ', k)


# crop val
iPath = os.path.join (iMainPath, 'val')
CheckDirectory (iPath)
aPath = os.path.join (aMainPath, 'val')
CheckDirectory (aPath)
for k in range(0, valN):
	Video2Frame (oPath, iPath, aPath, valFile, cropBound, k, capN_max, frameSkip, modeOr)
	print('val crop done, ', k)

# crop test
iPath = os.path.join (iMainPath, 'test')
CheckDirectory (iPath)
aPath = os.path.join (aMainPath, 'test')
CheckDirectory (aPath)
for k in range(0, testN):
	Video2Frame (oPath, iPath, aPath, testFile, cropBound, k, capN_max, frameSkip, modeOr)
	print('test crop done, ', k)

'''
for k in range(0,len(scen)):
	wPath = os.path.join(mainPath, scen[k])
	print(wPath)
	CheckDirectory (wPath)

	for kk in range(0,videoN[k]):
		oPath = 'origin/' + scen[k] + '/video' + str(kk)
		wPath = os.path.join(wPath, '/video' + str(kk))
		#CheckDirectory (wPath)
'''
