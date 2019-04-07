# Automatic Traffic Red-Light Running Violation Detection
## Abstract
An automatic traffic red-light violation detection system was implemented, which may play a big role in transportation management in smart cities. The system mainly relies on modern computer vision techniques, which was implemented in OpenCV under Python environment. Mainly, the system consists of object detector and object tracker which work in an integrated manner in order to precisely keep position of the existing cars. The primary task of the system is to eventually indicate locations of violating vehicles. The output showed accurate results, as all violating vehicles were detected and distinguished precisely.

## Design Specification
In this project, the system was implemented to accept the input as video of crossroad, which needed to observe and detect traffic red-light violations. System consists mainly of three components: violation line estimator, vehicles detector and vehicles tracker.

The figure below illustrates our proposed flow:

### Violation line estimation
A crucial part in building a system for vehicle violation detection is specifying the area for which a vehicle is considered to be in a violation state. Such task can be quite challenging and nearly unattainable due to the high levels of variability present in the input videos such as the depth of view, video capture perspective, location and distribution of traffic lights along with the specific description of the road. Consequently, it is clear that in order to obtain the desired region of violation, one must set direct assumptions which narrow down the generality of the problem and bring it closer to a technical level. Therefore, two main assumption regarding the input were made in order to gain knowledge about the violation region and proceed properly with the needed operations, these assumptions are: 
  • First, the region is to be represented as any vertical area within the video which surpasses a certain threshold horizontal line, referred to as the violation line, which splits the road into a regular and violating zone, as the main goal of the system evidentially becomes to determine the vertical location of this horizontal line. 
  • Second, in order to obtain clear info and indication towards the approximate vertical location of the violation line, it is required that a crosswalk exists between the regular and violating area, within the vicinity of the traffic light. The previously discussed assumptions are depicted in the following figure.

![Alt text](/lineDetection1.png?raw=true "Violation line approximation")

The figure shows the desired approximation of the violation line, as the location of the pedestrian crossway is the main element taken into consideration. Therefore, as a first step it was necessary to obtain clear locations of the crosswalk components. In order to obtain locations of crosswalk components, a sequence of processing steps was done, summarized as follows: 
  1. Convert the image into a greyscale in order to be used as an input to binary level thresholding. 

  2. Binarize the image based on an experimentally decided threshold value.

![Alt text](/lineDetection2.png?raw=true "Image thresholding")

  3. Perform a sequence of erosion and dilation operations in order to improve the components representing the crosswalk,      and also to limit the level of noise present in the image following the binarization.

![Alt text](/lineDetection3.png?raw=true "Dilation and Erosion")

  4. Following the output of dilation and erosion, it is now possible to detect contours in the binary image and show them      on the original frame, a contour is a closed curve of points or line segments, representing the boundaries of an object      in an image.

![Alt text](/lineDetection4.png?raw=true "Finding contours")

  5. As seen in the output in Figure 2-6, many contours give no information at all as they are just bounding segments of        white blobs. A filtering method is now needed in order to return the contours of the crosswalk components, which is done    first by taking into consideration the contours represented within a limited number of points and area, which lowers        randomness in the resulting contours.

![Alt text](/lineDetection5.png?raw=true "Contours filtering")

  6. The final step is to accept only contours with approximating points that describe a rectangular shape based on the        OpenCV polygon approximation.

![Alt text](/lineDetection6.png?raw=true "Rectangular contours filtering")

  7. For better visualization and analysis , bounding boxes for each contour are shown.
  Now that the crosswalk components have been detected, an extra step to avoid false positive contours is considered, as the   closest contour to the traffic light is found and is then considered as the component corresponding the crosswalk as a       whole. The vertical coordinate of this component is used to finally draw the horizontal violation line to be used in the     tracking logic and violation criteria.

![Alt text](/lineDetection7.png?raw=true "Crosswalk bounding boxes")
  
Now that the crosswalk components have been detected, an extra step to avoid false positive contours is considered, as the closest contour to the traffic light is found and is then considered as the component corresponding the crosswalk as a whole. The vertical coordinate of this component is used to finally draw the horizontal violation line to be used in the tracking logic and violation criteria.

![Alt text](/lineDetection8.png?raw=true "Finding closest box")

### Vehicle detection
At each five frames, a detection is done using YOLOv3 pretrained model on COCO dataset. The detection output is formulated by several steps, from filtering the bounding boxes with low confidence rate and filtering any bounding box that isn’t a vehicle to finally doing non-maximum suppression to the detected boxes, so that each vehicle has only one bounding box.

### Vehicle tracking
Keeping an accurate localization on each vehicle can be very tedious task, tracking the first frame vehicles is not enough, due to incoming and outgoing vehicles of the scope of the images. To resolve this, a merging between tracking and detection tasks is needed. The tracking of the vehicles is done at every frame, but every five frames a new detection occurs, then for each detected bounding box a measure of IoU (Intersection over Union) is done with the current tracking bounding boxes. If a detected box matches a tracking box with a relatively good percentage then it's the same box that in the tracking list, but the new detected bounding box has to be more accurate than the tracking one, so an adjustment operation occurs to the tracking object bounding box.


However, if there are detected boxes with no good matches with any of the tracking boxes, then they have to be a new incoming vehicle, when so, the tracking list is updated with those new vehicles, in contrast if there’s a bounding box that disappeared in the detection boxes, then it’s an outgoing car, so its tracker is removed. As mentioned above, at each iteration an update operation to the tracking list is done, when so, each bounding box is applied to a validator to check the violation, if it violates the red light, it’s tracking object is moved to the violated tracking list so it can be visualized in the output video.

### Violation detection
A vehicle violates if it satisfies two conditions, if its center y value crosses the violation line AND if the current status of the detected traffic light is red, when those two conditions present the vehicle violates the red light and its tracker is moved to the violated tracking list. When the traffic light is detected, recognizing its status can be done greedily using color histogram. Each pixel in the traffic light is mapped to either red, yellow/orange, green or other, after this the maximum summation of the pins is simply chosen as the status of the traffic light.

## System Evaluation
An important aspect in implementing a violation system is to thoroughly asses the outputs and results and reflect on the performance of the system in general based on common metrics. First, the system is to be evaluated on the processing time which will be categorized mainly into violation line approximation time, traffic light localization time and tracking and violation detection processing time (per frame). Finally, the metrics of average precision and average recall will be calculated for the input videos present and based on whether a vehicle is classified to be violating or not, namely, a false positive classification would be a non-violating vehicle being classified as violating, inversely, a false negative violation would be a violating vehicle being classified as non-violating.
### Processing Time Evaluation
The following table presents the average processing time for the metrics mentioned above following the testing of the input videos:

| Measurement                                   | Average Value (Seconds) |
| --------------------------------------------- | ----------------------- |
| Traffic Light Localization                    | 0.267                   |
| Violation Line Approximation                  | 1.933                   |
| Tracking and Violating Detection per Frame    | 0.294                   |


### Precision and Recall
Traffic surveillance systems in general have a high level of variability and dependency and are therefore prone to error and misclassification. Hence, it is necessary to calculate common measures such as average values of precision and recall in order to gain knowledge regarding system performance. In the implemented violation system, misclassification occurred rarely due to abnormal cases present in the input video First, a false positive misclassification occurred due to the violation line extension to a region beyond the scope of the target traffic light, which is a problem concerning the topology of the road and its corresponding lanes. The following figure depicts such misclassification:

|                | Violating | Non-Violating |
| ---------------|-----------|-------------- | 
| Violating      | 20        | 1             |
|  Non-Violating | 2         | 101           |


Such resultsare slight imperfections which result due to the high variability and randomness present in the input test video. It is worth noting that the process is completely automated in a manner where each component is non-parametrized as the system’s only input is the test video itself. Nevertheless, a total of 20 input videos were used to test the performance of the system with considering the classification output of each vehicle crossing the violation line under all conditions, the following is the resulting confusion matrix followed by precision, recall and F1-Score from outputs of the whole test set: 

|                | Precision | Recall | F1-Score  |
| ---------------|-----------|--------| --------- | 
| Violating      | 0.91      | 0.95   | 0.93      |
|  Non-Violating | 0.99      | 0.98   | 0.98      |


## Results

### Youtube: https://www.youtube.com/watch?v=7UjyifjnzJg

### Before and After shots
![Alt text](/output_appendix1.png?raw=true "Output Appendix 1")
![Alt text](/output_appendix2.png?raw=true "Output Appendix 2")

Done by:
- Ahmad Yahya ( @AhmadYahya97 )
- Ihab Abdelkareem ( @ihababdelkareem )
- Osama Al-fakhouri


