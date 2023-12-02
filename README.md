# Foucault_KnifeEdge_Mirror

**Foucault Knife Edge Shadowgram Analyzer for a primary mirror for Dobsonian telescopes**

This program was inspired by a discussion with reference to the following thread. https://www.cloudynights.com/topic/598980-foucault-unmasked-new-foucault-test-software/

**What is a Shadowgram?**

Stellafane provides a simple and excellent explanation below:

https://stellafane.org/tm/atm/test/shadowgrams.html

## Dependencies and Installation

## Windows 11/12

#### Pre-requisites 

1. Install Python 3 (version 3.11 or greater preferred) from windows software installation search tool.
2. Install Git 64-bit version via https://git-scm.com/download/win
3. Open a new Windows command prompt.
 pip3 install opencv-python scipy matplotlib
4. `cd <user dir path>\Downloads`
5. `git clone https://github.com/enthusiasticgeek/Foucault_KnifeEdge_Mirror.git`


#### Run: (To be executed each time on PC booting)

1. Open Windows command prompt.
2. `cd <user dir path>\Downloads\Foucault_KnifeEdge_Mirror` 
**Optional** (to get the latest software updates): `git pull`
3. Usage: `python3 ./src\Foucault_KnifeEdge_Shadowgram_Analyzer.py images\1.jpg`
4. Replace 1.jpg with your 640x480 resolution or slightly smaller image.
5. You may pass more flags as deemed necessary e.g., -bt 20 for brightness tolerance. README.md contains the complete list of flags.
6. Repeat steps 3 through 5 as needed.

#### Ubuntu/Debian 

**Python 3.8+ installation**

**Tested on Ubuntu 18.04/20.04 LTS**

#### Pre-requisites

pip3 install opencv-python scipy matplotlib

***One may use virtualenv https://virtualenv.pypa.io/en/latest/ to create isolated Python environments***

#### Run: (To be executed each time on PC booting)

See example below:

## Usage

**Example of usage on Ubuntu:** 

./resize_image.sh \<image file\>

./Foucault_KnifeEdge_Shadowgram_Analyzer.py [-h] [-d MINDIST] [-p1 PARAM1] [-p2 PARAM2] [-minR MINRADIUS] [-maxR MAXRADIUS] [-cc CONSIDERCONTOURS] [-dc DRAWCONTOURS] [-dnc DRAWNESTEDCONTOURS]
                                                 [-dr DRAWCIRCLES] [-bt BRIGHTNESSTOLERANCE] [-dwp DISPLAYWINDOWPERIOD] [-spnc SKIPPIXELSNEARCENTER] [-svi SAVEIMAGE] [-svp SAVEPLOT]
                                                 [-cmt CLOSESTMATCHTHRESHOLD] [-fli SHOWFLIPPEDIMAGE]
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

**Note:** Use 640x480 or smaller resolution images for faster processing (although the program will attempt to resize the images while still maintaning the aspect ratio). ImageMagick should do the trick on Linux. `convert input_image.jpg -resize 640x480 output_image.jpg`

**Note:** A CSV output file is created with x-cordinate, y-coordinate, average intensity, distance from x. Column 1 is all points to the left of center of mirror and Column 2 is to the right of the center of mirror. Distance values are in pixels and the intensity value per pixel varies between 0 (darkest) - 255 (brightest)

**Note:** One may try to vary the --brightnessTolerance value between 10 and 100 for the best results depending on the quality of the image. The default value of 20 suffices for most cases. 

**Note:** Set --displayWindowPeriod 0 if one doesn't want to automatically exit the program after 10 seconds.

**Note:** One may try to vary the --skipPixelsNearCenter value if not interested in calculating the intensity in the immediate neighbourhood of the center of the mirror. The default value of 40 suffices for most cases.


## Troubleshooting and FAQs

1. **Python script takes too long to execute and display plots and images! Can I tweak some parameters?**

   **Potential Fix:** The detector only works if it is fed a decent quality image. Given this is true (a) Check the value of brightness toleance and try lowering it (e.g. 10) since wider the range - most pixels with similar intensities will lie between the range (b) Check the image resolution and reduce to 640x480.

2. **Python script crashes with segfault or exits with other errors! What could be the reason?**

   **Potential Fix:** Either comment or uncomment `#matplotlib.use('tkagg')` line at the top of the Python script (as applicable). Some Operating system packages may or may not need this for matplotlib and opencv to work well together. Also ensure the image file exists at the location that is passed as an argument to the Python script.

3. **Python script is unable to detect the mirror! How do I correct this?**

   **Potential Fix:** Adjust p1, p2, minR, maxR, minDist parameters if necessary. The default values p1 (20), p2 (60), minR (10), maxR (0), minDist (5) are sufficient for majority of the cases. These parameters are critical to detect mirror from the photo using HoughCircularTransform function. 

**Credits:**

1. Guy Brandenburg, President, National Capital Astronomers (NCA) Amateur Telescope Making (ATM) workshop <http://www.capitalastronomers.org/>.
2. Alan Tarica, Mentor at ATM workshop.
