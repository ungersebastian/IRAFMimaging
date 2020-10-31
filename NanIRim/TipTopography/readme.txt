Steps for running program:
	Step 1: run correlationTopoImage.py
		we should need to make a good correlation between topo images and get their relative shift	
	Step 2: run dataFormatTransfer.py
		we need to change the shift in this program according to the shift we calcaulated in step1. The lines should be changed is from 58-64.
		we also need to change the saving directory. The lines from 115-119.
	Step 3: run getPointsPosition.py
		we need to change directory and keep it same as what we changed in step 2.
		we also need to change the saving directory. The line 225.
	Step 4: run drawSlopePointsOnToPo.py 
		we need to change directory and keep it same as what we changed in step 3. The line 51.
		we also need to change directory and keep it same as what we changed in step 2. The lines from 61-64.
	Then we could get the figure.
	if we run getPointsPositionVerticalCase.py in step 3, we should run corresponding program drawSlopePointsOnToPoVerticalCase.py in step 4.
	
Other programs:
	plotLineCuts.py:
		will draw and save all line cuts in one time
	posterImageLineCut.py: 
		will plot the figure of line cut in our poster. before we use it, we need to change directory.
	saveImageByNIP.py:
		tells how to save image by using NanoImagingPack.
	drawTwoToPoImage.py:
		the topo images shown on our poster
	drawTwoPiFMImage.py:
		the PiFM images shown on our poster
	viewFigures.py:
		method to view image by using NanoImagingPack.
	draw3DFigureWithNormalization.py:
		draw 3D figure, but the result is hard for our eyes to distinguish
	useGuiDisplayLineCuts.py :
		we could change parameters and draw image of aribitary slice in aribitary interval