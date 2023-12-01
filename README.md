# Foucault_KnifeEdge_Mirror

**Foucault Knife Edge Shadowgram Analyzer for a primary mirror for Dobsonian telescopes**

This program was inspired by a discussion with reference to the following thread. https://www.cloudynights.com/topic/598980-foucault-unmasked-new-foucault-test-software/

## Dependencies

**Python 3.8+ installation**

**Tested on Ubuntu 18.04/20.04 LTS**

pip3 install opencv-python scipy

***One may use virtualenv https://virtualenv.pypa.io/en/latest/ to create isolated Python environments***

## Usage

**Usage:** 

./Foucault_KnifeEdge_Shadowgram_Analyzer.py [-h] [-d MINDIST] [-p1 PARAM1] [-p2 PARAM2] [-minR MINRADIUS] [-maxR MAXRADIUS] [-cc CONSIDERCONTOURS] [-dc DRAWCONTOURS] [-dnc DRAWNESTEDCONTOURS]
                                                 [-dr DRAWCIRCLES] [-bt BRIGHTNESSTOLERANCE] [-dwp DISPLAYWINDOWPERIOD] [-spnc SKIPPIXELSNEARCENTER] [-svi SAVEIMAGE] [-svp SAVEPLOT]
                                                 [-cmt CLOSESTMATCHTHRESHOLD]
                                                 filename

**Usage example 1:** 

./src/Foucault_KnifeEdge_Shadowgram_Analyzer.py images/1.jpg --brightnessTolerance 20 --drawContours 1

**Final Analyzed Output**

![alt text]( https://github.com/enthusiasticgeek/Foucault_KnifeEdge_Mirror/blob/main/images/saved_gray_image.png "example output")

**Plot**

![alt text]( https://github.com/enthusiasticgeek/Foucault_KnifeEdge_Mirror/blob/main/images/saved_plot.png "example output")

Image with debug feature turned on (--drawContours 1)

![alt text]( https://github.com/enthusiasticgeek/Foucault_KnifeEdge_Mirror/blob/main/images/saved_debugging_image.png "example output")

**Usage example 2:** 

**2D to 3D intensity plot with local maxima.**

./src/intensity_image_2d_to_3d.py images/1.jpg

![alt text]( https://github.com/enthusiasticgeek/Foucault_KnifeEdge_Mirror/blob/main/images/3d_intensity_plot1.png "example output")

**Usage example 3:** 

**DFT Analysis**

./src/DFT_analysis.py images/1.jpg

![alt text]( https://github.com/enthusiasticgeek/Foucault_KnifeEdge_Mirror/blob/main/images/DFT_analysis.png "example output")


## Limitations/Recommendations

**Note:** Use 640x480 or smaller resolution images for faster processing (although the program will attempt to resize the images while still maintaning the aspect ratio).

**Note:** Adjust p1, p2, minR, maxR, minDist parameters if necessary. The default values p1(20), p2(60), minR(10), maxR (0), minDist(5) are sufficient for majority of the cases. These parameters are critical to detect mirror from the photo using HoughCircularTransform function.

**Note:** A CSV output file is created with x-cordinate, y-coordinate, average intensity, distance from x. Column 1 is all points to the left of center of mirror and Column 2 is to the right of the center of mirror. Distance values are in pixels and the intensity value per pixel varies between 0 (darkest) - 255 (brightest)

**Note:** One may try to vary the --brightnessTolerance value between 10 and 100 for the best results depending on the quality of the image. The default value of 20 suffices for most cases. 

**Note:** Set --displayWindowPeriod 0 if one doesn't want to automatically exit the program after 10 seconds.

**Note:** One may try to vary the --skipPixelsNearCenter value if not interested in calculating the intensity in the immediate neighbourhood of the center of the mirror. The default value of 40 suffices for most cases.

**Credits:**

1. Guy Brandenburg, President, National Capital Astronomers (NCA) Amateur Telescope Making (ATM) workshop <http://www.capitalastronomers.org/>.
2. Alan Tarica, Mentor at ATM workshop.
