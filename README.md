# Speed_Prediction
This is for a project I am working on in my Neural Network course. This project was inpsired from a comma.ai programming challange.

Your goal is to predict the speed of a car from a video.

- data/train.mp4 is a video of driving containing 20400 frames. Video is shot at 20 fps.
- data/train.txt contains the speed of the car at each frame, one speed on each line.
- data/test.mp4 is a different driving video containing 10798 frames. Video is shot at 20 fps.
The train.mp4 went through some image processing, by first extracting the frames into gray-scale images. Then the following was done:

- An index and space was taken,
- the index represents the frame we start at and the space is the frame after and before the index (ex. index=100, space=2 then we want frames index+space=102 and index-space = 98)
- I then took the subtraction between both frames and addition and sharpened the images to view the chnages more clearly between both frames.
- these extracted images are then placed in different folers to analyze and then pick what data we want to feed into the CNN.
