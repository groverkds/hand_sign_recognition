# hand_sign_recognition
A simple sign language recognizer using SVM and scikit-learn's adjusted_rand_score (basic correlation).

Steps to setup before you run:
1) install pandas package (pip3 install pandas)
2) install sklearn (pip3 install scikit-learn)
3) Create a folder for images and subfolders for train and test images.
4) run generate_image_features.py
5) run train.py

Now run the main.py file.


You can repeat 3,4 any number of times (at least once generate the classifier and feature set)
Comment out the code as per your requirement in main.py( for selecting video/test images/run on sample photos)

Folder name convention for training set:
images/train/A (contains all the photos for A's symbol,similar for other characters)
Folder name convention for test set:
images/train/A.jpg (Test for A's symbol,similar for other characters)

To get entire training set (80 mb): https://drive.google.com/drive/folders/0Bw239KLrN7zoNkU5elZMRkc4TU0?usp=sharing
Training set images were uploaded by Anmol Singh Jaggi.