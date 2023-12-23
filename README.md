# Foucault_KnifeEdge_Mirror

![alt text]( https://github.com/enthusiasticgeek/Foucault_KnifeEdge_Mirror/blob/main/fkesa.png "example output")

**Foucault Knife Edge Shadowgram Analyzer (FKESA) for a primary mirror [Dobsonian telescopes]**

This program was inspired by a discussion with reference to the following thread. https://www.cloudynights.com/topic/598980-foucault-unmasked-new-foucault-test-software/

**What is a Shadowgram?**

Stellafane provides a simple and an excellent explanation below:

https://stellafane.org/tm/atm/test/shadowgrams.html

## Dependencies and Installation

## Windows 11/12

#### Pre-requisites: 

1. Install Python 3 (version 3.11 or greater preferred) from windows software installation search tool.
2. Install Git 64-bit version via https://git-scm.com/download/win
3. Open a new Windows command prompt.

	`pip3 install opencv-python scipy matplotlib PySimpleGUI Pillow`

4. `cd <user dir path>\Downloads`
5. `git clone https://github.com/enthusiasticgeek/Foucault_KnifeEdge_Mirror.git`

#### Run: (To be executed every time)

1. Open Windows command prompt.
2. `cd <user dir path>\Downloads\Foucault_KnifeEdge_Mirror` 
**Optional** (to get the latest software updates): `git pull`
3. ~Resize the image e.g. 1.jpg to 640x480 and~ copy the same in `images` folder. Adjust path and image filename in the following command line to where the image file is located if other than `images` folder.
4. Usage: `python3 ./src\Foucault_KnifeEdge_Shadowgram_Analyzer.py images\1.jpg`
5. ~Replace 1.jpg with your 640x480 resolution or slightly smaller image.~
6. You may pass more flags as deemed necessary e.g., -bt 20 for brightness tolerance. README.md contains the complete list of flags.
7. Repeat steps 3 through 5 as needed.

## Linux

#### Ubuntu/Debian 

Tested on Ubuntu 18.04/20.04 LTS

#### Pre-requisites:

	# Python 3.8+ installation (Python 3.11+ preferred)

and execute

	pip3 install numpy opencv-python scipy matplotlib PySimpleGUI Pillow

or optionally with `-m` flag turned on https://stackoverflow.com/questions/50821312/what-is-the-effect-of-using-python-m-pip-instead-of-just-pip

        python -m pip install numpy opencv-python scipy matplotlib PySimpleGUI Pillow
	
On Ubuntu20.04 LTS, one may have to execute the following as well.

	sudo apt install python3-pil.imagetk

***One may use virtualenv https://virtualenv.pypa.io/en/latest/ to create isolated Python environments***

#### Run: (To be executed every time)

See example below:

## Usage

**Example of usage on Ubuntu:** 

~Use ImageMagick (Linux) to resize the image to 640x480 resolution first (other smaller resolutions may also work).~

	./src/resize_image.sh <image filename>

Then execute the following (replacing correct parameters as deemed necessary)

	./src/Foucault_KnifeEdge_Shadowgram_Analyzer.py [-h] [-dir FOLDER] [-d MINDIST] [-p1 PARAM1] [-p2 PARAM2] [-minR MINRADIUS] [-maxR MAXRADIUS] [-cc CONSIDERCONTOURS] [-dc DRAWCONTOURS] [-dnc DRAWNESTEDCONTOURS]
                                                 [-dr DRAWCIRCLES] [-bt BRIGHTNESSTOLERANCE] [-dwp DISPLAYWINDOWPERIOD] [-spnc SKIPPIXELSNEARCENTER] [-svi SAVEIMAGE] [-svf SAVEFLIPPEDIMAGE]
                                                 [-svc SAVECONTOURSIMAGE] [-svcrp SAVECROPPEDIMAGE] [-svp SAVEPLOT] [-sip SHOWINTENSITYPLOT] [-spl SHOWPLOTLEGEND] [-cmt CLOSESTMATCHTHRESHOLD]
                                                 [-fli SHOWFLIPPEDIMAGE] [-lai LISTALLINTESITIES] [-rpil RESIZEWITHPILLOW] [-mdia MIRRORDIAMETERINCHES] [-mfl MIRRORFOCALLENGTHINCHES]
                                                 [-rfc RETRYFINDMIRROR] [-gmc GAMMACORRECTION] [-usci USESIGMACLIPPEDIMAGE]
                                                 <image filename>

The parameters (flags) description and their respective default values are as follows:

	'-dir', '--folder', default='', help='Folder name/path (default: {filename}_output)'
	'-d', '--minDist', type=int, default=50, help='Minimum distance between detected circles
	'-p1', '--param1', type=int, default=30, help='First method-specific parameter
	'-p2', '--param2', type=int, default=60, help='Second method-specific parameter
	'-minR', '--minRadius', type=int, default=10, help='Minimum circle radius
	'-maxR', '--maxRadius', type=int, default=0, help='Maximum circle radius
	'-cc', '--considerContours', type=int, default=0, help='Draw intensity only accounting for contours of the shadowgram. This makes analysis more detailed. Default value is 0
	'-dc', '--drawContours', type=int, default=0, help='Draw contours
	'-dnc', '--drawNestedContours', type=int, default=0, help='Draw Nested contours
	'-dr', '--drawCircles', type=int, default=1, help='Draw mirror circle(s)
	'-bt', '--brightnessTolerance', type=int, default=20, help='Brightness tolerance value for intensity calculation. Default value is 20
	'-dwp', '--displayWindowPeriod', type=int, default=10000, help='Display window period 10 seconds. Set to 0 for an infinite window period.
	'-spnc', '--skipPixelsNearCenter', type=int, default=40, help='Skip the pixels that are too close to the center of the mirror for intensity calculation. Default value is 40
	'-svi', '--saveImage', type=int, default=1, help='Save the Analysis Image on the disk (value changed to 1). Default value is 1
	'-svf', '--saveFlippedImage', type=int, default=1, help='Save the Flipped Image on the disk (value changed to 1). Default value is 1
	'-svc', '--saveContoursImage', type=int, default=0, help='Save the Contour Image on the disk. Default value is 0
	'-svcrp', '--saveCroppedImage', type=int, default=1, help='Save the Cropped Image on the disk (value changed to 1). Default value is 1
	'-svp', '--savePlot', type=int, default=1, help='Save the Analysis Plot on the disk (value changed to 1). Default value is 1
	'-spl', '--showPlotLegend', type=int, default=0, help='Show plot legend. Default value is 0
	'-sip', '--showIntensityPlot', type=int, default=1, help='Show the Analysis Plot (value changed to 1). Default value is 1'
	'-cmt', '--closestMatchThreshold', type=int, default=2, help='Threshold value that allows it to be considered equal intensity value points. Default value is 3
	'-fli', '--showFlippedImage', type=int, default=0, help='Show absolute difference, followed by flipped and superimposed cropped image. Default value is 0
	'-lai', '--listAllIntesities', type=int, default=1, help='List all Intensities data regardless of matching intensities. It created two CSV files - one for all data (this flag) and another matching data only. Default value is 1
	'-rpil', '--resizeWithPillow', type=int, default=0, help='Resize with Pillow instead of OpenCV. Default value is 0'
	'-mdia', '--mirrorDiameterInches', type=int, default=6, help='Mirror diameter in inches. Default value is 6'
	'-mfl', '--mirrorFocalLengthInches', type=float, default=48, help='Mirror Focal Length in inches. Default value is 48.0'
	'-rfc', '--retryFindMirror', type=int, default=1, help='Adjust Hough Transform search window (adaptive) and attempt to find Mirror. default 1'
	'-gmc', '--gammaCorrection', type=float, default=2.2, help='Adjust image gamma correction. default 2.2'
	'-usci', '--useSigmaClippedImage', type=int, default=0, help='Sigma Clipped Image. default 0'


**Usage example 1.a:** 

	./src/Foucault_KnifeEdge_Shadowgram_Analyzer.py images/1.jpg --brightnessTolerance 20 --drawContours 1

**Final Analyzed Output**

**Note:** The distance values are in pixels unless otherwise specified. Max intensity value is 255 and min intensity value is 0. Please change `--mirrorDiameterInches` flag to a value to match one's mirror diameter as the default value is `6.0` inches e.g. `-mdia 4.25`. Similarly, please change `--mirrorFocalLengthInches` flag to a value to match one's mirror focal length as the default value is `48.0` inches e.g. `-mfl 20.50`. In the following example we have set `--gammaCorrection 2.2` for demonstration purposes but it is not necessary in most cases.

![alt text]( https://github.com/enthusiasticgeek/Foucault_KnifeEdge_Mirror/blob/main/images/saved_gray_image.png "example output")

**Plot**

![alt text]( https://github.com/enthusiasticgeek/Foucault_KnifeEdge_Mirror/blob/main/images/saved_plot.png "example output")

Image with debug features turned on (`--drawContours 1` and `--drawNestedContours 1`)

![alt text]( https://github.com/enthusiasticgeek/Foucault_KnifeEdge_Mirror/blob/main/images/saved_debugging_image.png "example output")

Image with flip and superimpose feature turned on (`-fli 1`)

![alt text]( https://github.com/enthusiasticgeek/Foucault_KnifeEdge_Mirror/blob/main/images/phi_flipped.png "example output")

The following CSV files are generated in the same folder where image file is situated. e.g. for `1.jpg`, `1.jpg.data.csv` has the following columns `X, Y, INTENSITY` [All data points], `1.jpg.zones.csv` has the following columns `NULL ZONE LEFT [inches], NULL ZONE LEFT [mm], INTENSITY LEFT, NULL ZONE RIGHT [inches], NULL ZONE RIGHT [mm], INTENSITY RIGHT, X [pixels], Y [pixels], RADIUS [pixels]` [More information on zones from matches], `1.jpg.csv` has the following columns `Less Than X1 (LEFT) (x [pixels], y [pixels], intensity [0-255], distance from X1 [pixels])	Greater Than X1 (RIGHT) (x [pixels], y [pixels], intensity [0-255], distance from X1 [pixels])` [Only matches from all data points]

	-rw-rw-r-- 1 enthusiasticgeek enthusiasticgeek   3236 Dec 10 16:10 1.jpg.data.csv
	-rw-rw-r-- 1 enthusiasticgeek enthusiasticgeek    254 Dec 10 16:10 1.jpg.zones.csv
	-rw-rw-r-- 1 enthusiasticgeek enthusiasticgeek    403 Dec 10 16:10 1.jpg.csv

**Update**

**Usage example 1.b:** 

A new way to calculate using zones specifying ROI angle (zones in a slice of a pie) is now available!!! 

	./src/FKESA_v2.py images/1.jpg

The parameters (flags) description and their respective default values are as follows:

	'filename', help='Path to the image file'
	'-dir', '--folder', default='', help='Folder name/path (default: {filename}_output)'
	'-d', '--minDist', type=int, default=50, help='Minimum distance between detected circles. Default 50'
	'-p1', '--param1', type=int, default=20, help='First method-specific parameter. Default 30'
	'-p2', '--param2', type=int, default=60, help='Second method-specific parameter. Default 60'
	'-minR', '--minRadius', type=int, default=10, help='Minimum circle radius. Default 10'
	'-maxR', '--maxRadius', type=int, default=0, help='Maximum circle radius. Default 0'
	'-dwpz', '--displayWindowPeriodZones', type=int, default=100, help='Maximum period to wait in milliseconds between displaying zones. Default 100 milliseconds'
	'-bt', '--brightnessTolerance', type=int, default=10, help='Brightness Tolerance. Default 10'
	'-rad', '--roiAngleDegrees', type=int, default=10, help='ROI angle degrees. Default 10'
	'-z', '--Zones', type=int, default=50, help='Number of zones [30 to 50]. Default 50'
	'-szfc', '--skipZonesFromCenter', type=int, default=10, help='Skip Number of zones from the center of the mirror. Default 10'
	'-rfc', '--retryFindMirror', type=int, default=1, help='Adjust Hough Transform search window (adaptive) and attempt to find Mirror. default 1'
	'-mdia', '--mirrorDiameterInches', type=float, default=6, help='Mirror diameter in inches. Default value is 6.0'
	'-mfl', '--mirrorFocalLengthInches', type=float, default=48, help='Mirror Focal Length in inches. Default value is 48.0'
	'-svi', '--saveImage', type=int, default=1, help='Save the Analysis Image on the disk (value changed to 1). Default value is 1'
	'-dwp', '--displayWindowPeriod', type=int, default=10000, help='Display window period 10 seconds. Set to 0 for an infinite window period.'
	'-svp', '--savePlot', type=int, default=1, help='Save the Analysis Plot on the disk (value changed to 1). Default value is 1'
	'-spl', '--showPlot', type=int, default=1, help='Display the Analysis Plot (value changed to 1). Default value is 1'
	'-gmc', '--gammaCorrection', type=float, default=0, help='Adjust image gamma correction. Typical correction value is 2.2. default 0'
	'-fli', '--showFlippedImage', type=int, default=0, help='Show absolute difference, followed by flipped and superimposed cropped image. Default value is 0'
	'-svf', '--saveFlippedImage', type=int, default=1, help='Save the Flipped Image on the disk (value changed to 1). Default value is 1'


![alt text]( https://github.com/enthusiasticgeek/Foucault_KnifeEdge_Mirror/blob/main/images/1.fkesa_v2.analysis.jpg "example output")

**Usage example 2:** 

**2D to 3D intensity plot with local maxima.**

	./src/intensity_image_2d_to_3d.py images/1.jpg

![alt text]( https://github.com/enthusiasticgeek/Foucault_KnifeEdge_Mirror/blob/main/images/3d_intensity_plot1.png "example output")

**Usage example 3:** 

**DFT Analysis**

	./src/DFT_analysis.py images/1.jpg

The generated output image is also saved to the disk.

	magnitude_spectrum_<timestamp>.png

![alt text]( https://github.com/enthusiasticgeek/Foucault_KnifeEdge_Mirror/blob/main/images/DFT_analysis.png "example output")


## Limitations/Recommendations

**Note:** ~Use 640x480 or smaller resolution images for faster processing (although the program will attempt to resize the images while still maintaning the aspect ratio). ImageMagick should do the trick on Linux. `convert input_image.jpg -resize 640x480 output_image.jpg`~. Fixed image scaling operation.

**Note:** A CSV output file is created with x-cordinate, y-coordinate, average intensity, distance from x. Column 1 is all points to the left of center of mirror and Column 2 is to the right of the center of mirror. Distance values are in pixels and the intensity value per pixel varies between 0 (darkest) - 255 (brightest)

**Note:** One may try to vary the `--brightnessTolerance` value between 10 and 100 for the best results depending on the quality of the image. The default value of 20 suffices for most cases. 

**Note:** Set `--displayWindowPeriod 0` if one doesn't want to automatically exit the program after 10 seconds.

**Note:** One may try to vary the `--skipPixelsNearCenter` value if not interested in calculating the intensity in the immediate neighbourhood of the center of the mirror. The default value of 40 suffices for most cases.


## Troubleshooting and FAQs

1. **Python script takes too long to execute and display plots and images! Can I tweak some parameters?**

   **Potential Fix:** The detector only works if it is fed a decent quality image. Given this is true (a) Check the value of brightness tolerance and try lowering it (e.g. 10) since wider the range - most pixels with similar intensities will lie between the range (b) Check the image resolution and reduce to 640x480 especially if running the program over the following platforms 1. Windows Subsystem for Linux (WSL) https://learn.microsoft.com/en-us/windows/wsl 2. Docker Container https://www.docker.com/resources/what-container/ 3. VirtualBox https://www.virtualbox.org/ - See the resize image section below if using Windows 11/12.

2. **Python script crashes with segfault or exits with other errors! What could be the reason?**

   **Potential Fix:** Either comment or uncomment `#matplotlib.use('tkagg')` line at the top of the Python script (as applicable). Some Operating system installations setup may or may not need this for matplotlib and opencv to work well together. Also ensure the image file exists at the location that is passed as an argument to the Python script.

3. **Python script is unable to detect the mirror! How do I correct this?**

   **Potential Fix:** Adjust p1, p2, minR, maxR, minDist parameters if necessary. The default values p1 (20), p2 (60), minR (10), maxR (0), minDist (5) are sufficient for majority of the cases. These parameters are critical to detect mirror from the photo using HoughCircularTransform function. 

4. **How may I resize the images on Windows 11/12 since the supplied ImageMagick resize images bash script only works on Linux? Note: This is not necessary in most cases.**

   **Potential Fix:** There are numerous free software tools (and their respective documentation) available online that may help one accomplish this task (a) GIMP https://www.gimp.org/downloads/ (b) IrfanView https://www.irfanview.com/ (c) FastStone https://www.faststone.org/FSResizerDownload.htm . The following URL contains some good tips https://www.makeuseof.com/resize-images-windows-11/

**Credits:**

1. Guy Brandenburg, President, National Capital Astronomers (NCA) Amateur Telescope Making (ATM) workshop <http://www.capitalastronomers.org/>.
2. Alan Tarica, Mentor at ATM workshop.
