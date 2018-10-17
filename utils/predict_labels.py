import cv2
import numpy as np
from utils.load_data import process_image
import pickle
from sklearn.metrics import adjusted_rand_score
import sys
import os
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn import metrics
import pandas as pd
ROOT_DIR = os.path.dirname(__file__)
PARENT_DIR = os.path.dirname(__file__) + '/' + str(os.pardir)
sys.path.append(PARENT_DIR) # append parent directory to use db_util module

def predict_label_correlation(frame):
	input_data = process_image(frame)
	with open('data/train.pickle', 'rb') as handle:
		train = pickle.load(handle)
	correlation={}
	best_value = 0 
	found=''
	for key in train.keys():
	
		acc_list = []
		for img in train[key]:
			acc_list.append(adjusted_rand_score(img,input_data))
			
		correlation[key]=max(acc_list)				 
		if correlation[key]>best_value:
			best_value=correlation[key]
			found = key
	
	return found

def predict_label_svm(frame):
	input_data = process_image(frame)
	with open('data/classifier_SVM.pickle', 'rb') as handle:
		clf = pickle.load(handle)
	with open('data/dataframe.pickle', 'rb') as handle:
		df = pickle.load(handle)
	labels = list(df.columns)
	input_list = list(input_data)
	input_list.append(0)
	i_df = pd.DataFrame.from_records([tuple(input_list)], columns=labels)
	a = clf.predict(i_df)
	return chr(a[0]+65)