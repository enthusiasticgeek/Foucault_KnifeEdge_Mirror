# Foucault_KnifeEdge_Mirror

**Foucault Knife Edge Shadowgram Analyzer for a Mirror for Dobsonian telescopes**

This program was inspired from the following thread. https://www.cloudynights.com/topic/598980-foucault-unmasked-new-foucault-test-software/

**Usage:** 

Foucault_KnifeEdge_Shadowgram_Analyzer.py [-h] [--minDist MINDIST] [--param1 PARAM1] [--param2 PARAM2] [--minRadius MINRADIUS] [--maxRadius MAXRADIUS] [--canTolerance CANTOLERANCE]
                                                 [--drawContours DRAWCONTOURS] [--drawNestedContours DRAWNESTEDCONTOURS] [--drawCircles DRAWCIRCLES] [--brightnessTolerance BRIGHTNESSTOLERANCE]
                                                 [--displayWindowPeriod DISPLAYWINDOWPERIOD]
                                                 filename

**Usage example:** 

./src/Foucault_KnifeEdge_Shadowgram_Analyzer.py images/1.jpg --brightnessTolerance 20 --drawContours 1

Final Analyzed Output

![alt text]( https://github.com/enthusiasticgeek/Foucault_KnifeEdge_Mirror/blob/main/images/saved_gray_image.png "example output")

Image with debug feature turned on (--drawContour 1)

![alt text]( https://github.com/enthusiasticgeek/Foucault_KnifeEdge_Mirror/blob/main/images/saved_debugging_image.png "example output")

2D to 3D intensity plot with local maxima.

![alt text]( https://github.com/enthusiasticgeek/Foucault_KnifeEdge_Mirror/blob/main/images/3d_intensity_plot.png "example output")

**Note:** Use 640x480 or smaller resolution images for faster processing (although the program will attempt to resize the images while still maintaning the aspect ratio

**Note:** A CSV output file is created with x-cordinate, y-coordinate, average intensity, distance from x. Column 1 is all points to the left of center of mirror and Column 2 is to the right of the center of mirror

**Credits:**

1. Guy Brandenburg, President, National Capital Astronomers (NCA) Amateur Telescope Making (ATM) workshop <http://www.capitalastronomers.org/>
2. Alan Tarica, Mentor at ATM
