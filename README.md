# EECS504_Lotscanai

Here is how our pipeline works:
1) Our frontened is the YOLO detector, which the code is contained in the lotscan_ai_YOLO python file. The output of this python script will receive training data and will output centroids and other bounding-box related data
2) Afterwards, we'd use the lotscan_ai_mapping_to_master python file to be able to map those centroids from (1) into a top-down view and "stitch"/"match" the view to the master-grid. Then, we would use these transformed informations to update our theoretical consumer app and/or database.
3) Our highlightingSpots file holds scripts that would help us to draw lines onto our hypothetical app's UI. It also contains some other classical CV functions we experimented with to try and find parking spots.  
*) The homography python script is used as a helper function to help rectify any perspective image into a top-down view. This is very useful for making draws or computations easier.

   Link to the trained YOLO keras model: https://drive.google.com/file/d/1lCB7t-2aP-LeWMmRzrdI3fMlWT-_jrus/view?usp=drive_link
