# hand_sign_recognition
A simple sign language recognizer using SVM and scikit-learn's adjusted_rand_score (basic correlation).

Sample video: https://youtu.be/f3pSbjmQBwM

Steps to setup before you run:
1) install pandas package (pip3 install pandas)
2) install sklearn (pip3 install scikit-learn)
3) Create a folder for 'images' and subfolders for 'TRAIN' and 'TEST' images.
4) Create a folder 'data'.
5) run generate_image_features.py
6) run train.py

Now run the main.py file.


You can repeat 3,4 any number of times (at least once generate the classifier and feature set)
Comment out the code as per your requirement in main.py( for selecting video/test images/run on sample photos)

Folder name convention for training set:
images/train/A (contains all the photos for A's symbol,similar for other characters)
Folder name convention for test set:
images/train/A.jpg (Test for A's symbol,similar for other characters)


I would suggest you to try with your own hand images for training & testing the model since different hands have different complexion, size, other characteristics.
To get entire training set (80 mb): https://drive.google.com/drive/folders/0Bw239KLrN7zoNkU5elZMRkc4TU0?usp=sharing
Training set images were uploaded by Anmol Singh Jaggi.
