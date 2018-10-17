import cv2
import numpy as np
from utils.load_data import load_data 
import os
import pickle
import pandas as pd
if __name__ == "__main__":
	train_dict,df = load_data('images/TRAIN')
	test_dict,_ = load_data('images/TEST',0)
	with open('data/train.pickle', 'wb') as handle:
		pickle.dump(train_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
	with open('data/test.pickle', 'wb') as handle:
		pickle.dump(test_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
	with open('data/dataframe.pickle', 'wb') as handle:
		pickle.dump(df, handle, protocol=pickle.HIGHEST_PROTOCOL)