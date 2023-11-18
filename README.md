# Foucault_KnifeEdge_Mirror
Foucault Knife Edge Shadowgram Analyzer for a Mirror for Dobsonian telescopes

**Usage:** Foucault_KnifeEdge_Shadowgram_Analyzer.py [-h] [--minDist MINDIST] [--param1 PARAM1] [--param2 PARAM2] [--minRadius MINRADIUS] [--maxRadius MAXRADIUS] [--canTolerance CANTOLERANCE]
                                                 [--drawContours DRAWCONTOURS] [--drawNestedContours DRAWNESTEDCONTOURS] [--drawCircles DRAWCIRCLES] [--brightnessTolerance BRIGHTNESSTOLERANCE]
                                                 [--displayWindowPeriod DISPLAYWINDOWPERIOD]
                                                 filename

**Usage example:** ./src/Foucault_KnifeEdge_Shadowgram_Analyzer.py images/1.jpg --brightnessTolerance 20 --drawContours 1

![alt text]( https://github.com/enthusiasticgeek/Foucault_KnifeEdge_Mirror/blob/main/images/saved_debugging_image.png "example output")

![alt text]( https://github.com/enthusiasticgeek/Foucault_KnifeEdge_Mirror/blob/main/images/saved_gray_image.png "example output")

**Note:** Use 640x480 or smaller resolution images for faster processing (although the program will attempt to resize the images while still maintaning the aspect ratio

**Note:** A CSV file is created with x-cordinate, y-coordinate, average intensity, distance from x. Column 1 is all points to the left of center of mirror and Column 2 is to the right of the center of mirror

