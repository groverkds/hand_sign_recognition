import os
import cv2
import numpy as np
import sys
import pandas as pd
ROOT_DIR = os.path.dirname(__file__)
PARENT_DIR = os.path.dirname(__file__) + '/' + str(os.pardir)
sys.path.append(PARENT_DIR) # append parent directory to use db_util module

from utils.image_transformation import transform 

def load_data(path,train_flag=1):
	if train_flag==1:
		return load_train_data(path)
	elif train_flag==0:
		return load_test_data(path),None
	else:
		raise Exception('value of train_flag should be either 0(test data) or 1(train data)')

def load_test_data(path):
	list_files = os.listdir(path)
	image_dict = {}
	for file_name in list_files:
		im = cv2.imread(path+'/'+file_name)
		tr_im = process_image(im)
		image_dict[file_name.split('.')[0]]=tr_im
	
	return image_dict

def load_train_data(path):
	labels = ['p'+str(i+1) for i in range(10000)]
	labels.append('alphabet')
	list_tuples = []
	list_folders = os.listdir(path)
	train_dict={}
	for folder in list_folders:
		list_files = os.listdir(path+'/'+folder)
		train_dict[folder]=[]
		for file_name in list_files:
			im = cv2.imread(path+'/'+folder+'/'+file_name)
			tr_im = process_image(im)
			train_dict[folder].append(tr_im)
			temp = list(tr_im)
			temp.append(ord(folder)-65)
			list_tuples.append(tuple(temp))
	df = pd.DataFrame.from_records(list_tuples, columns=labels)
	return train_dict,df

def process_image(img):
	return transform(img).flatten()