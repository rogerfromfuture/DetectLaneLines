#**Finding Lane Lines on the Road**


### Reflection

###1. How the current pipeline works
####To get the raw hough lines and process images, it's consisted of the following steps.
1) Covert the image to grayscale
2) Apply gaussian blurring on the image 1)
3) Apply canny effect on the image from 2)
4) Define the region of interest on top of 3)
5) Get the hough lines on 4) to final overlay image
5) Use weight image function to overlay 5) on top of original image

The processed images are as follows
#####Solid white curve
![Solid white curve](https://raw.githubusercontent.com/rogerfromfuture/DetectLaneLines/master/processed/processed_solidWhiteCurve.jpg)

#####Solid white right
![Solid white right](https://raw.githubusercontent.com/rogerfromfuture/DetectLaneLines/master/processed/processed_solidWhiteRight.jpg)

#####Solid yellow curve
![Solid yellow curve](https://raw.githubusercontent.com/rogerfromfuture/DetectLaneLines/master/processed/processed_solidYellowCurve.jpg)

#####Solid yellow curve 2
![Solid yellow curve 2](https://raw.githubusercontent.com/rogerfromfuture/DetectLaneLines/master/processed/processed_solidYellowCurve2.jpg)

#####White car lane switch
![Solid car lane switch](https://raw.githubusercontent.com/rogerfromfuture/DetectLaneLines/master/processed/processed_whiteCarLaneSwitch.jpg)

#### Now in order to process the video, use the moviepy and feed the same process image function to the fl_image
In order to draw a single line on the left and right lanes, I added a new function called draw_one_line and apply to the
left and right lane. The hough lines function instead call this draw_one_line function
the algorithm is find the min X, Y coordinates for left and right lane, and also calculate the 50 percentile slope 
of all the hough lines respectively for left and right lane. Once the slope for left and right lanes are calculated
The top X,Y of the coordinates are figured so they could be drawn

I provided two videos for each white and yellow videos, one is with straight line the other is with raw hough lines
in the processed folder
raw lines:
https://raw.githubusercontent.com/rogerfromfuture/DetectLaneLines/master/processed/white_raw_houghlines.mp4
https://raw.githubusercontent.com/rogerfromfuture/DetectLaneLines/master/processed/yellow_raw_houghlines.mp4

Straight lines
https://raw.githubusercontent.com/rogerfromfuture/DetectLaneLines/master/processed/white.mp4
https://raw.githubusercontent.com/rogerfromfuture/DetectLaneLines/master/processed/yellow.mp4

###2. Identify potential shortcomings with my current pipeline
1) If the slope is not by average change dramatically from frame to frame then in the video it will look jittering
2) The current implementation I believe not gonna work well with left or right turns given the region is hardcoded

###3. Suggest possible improvements to your pipeline
The improvement to 1) could be save the average slope of past frames and then put that factor into calculation of the slope in the new frame
and by doing so, the jittering will be less

The improvement to 2) I can think of is since in the turn there will not only be straigh lines but be curves, to do it
maybe needs to figure out how to draw curve at certain point