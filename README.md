# Face-Mask-Detection!

When the video stream starts the camera begins reading the stream and then parses it frame by frame.
When a single frame is captured it is taken to the detect function where firstly it is checked if a face is visible. 
That is done by the pre-made models included in the project. If a face is detected then all the irrelevant background is cropped and only the face part is kept.
Meanwhile the whole frame is shown to the user with a rectangle hovering over the face, the rectangle would be red or green according to the classification made by the model.
Then the facial part is sent to the prediction function where first it is reshaped to 224*224*1 because our model was trained with images of these dimensions and the model won’t work with images of other dimensions.
After that is complete the prediction gives the output of either ‘Mask’ or ‘No Mask’.Then after getting that result we give the rectangle a colour and the correct label and the confidence given by the prediction.
After that is done, we iterate indefinitely and if for continuous 50 frames the output remains the same, we save the last frame with the output and then send the email to our desired administrator.


![accesses](https://user-images.githubusercontent.com/60512613/187853600-1bc9503e-a2c7-4191-a8a4-aef26a191250.jpg)
