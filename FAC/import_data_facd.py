import pickle
import pandas as pd
import numpy as np
import os
import torch
import torchvision
import torchvision.transforms as transforms
from PIL import Image
import cv2

class ImportData(object):
	def __init__(self, input_shape, data_dir=None):
		self.input_shape = input_shape
		self.data_dir = data_dir


	def find_image(self, category, image_id):
	    for file in os.listdir("./FACD_image/"+ category):
	        if image_id in file:
	#             print("./FACD_image/"+ category + "/" + str(image_id) + ".jpg")
	            return "./FACD_image/"+ category + "/" + str(image_id) + ".jpg"


	def load_data(self, datasplit=None):
		pickle_1 = open("FACD_metadata/pairwise_comparison.pkl","rb")
		pickle_2 = open("FACD_metadata/image_score.pkl","rb")
		comparison_pickle = pickle.load(pickle_1)
		image_score = pickle.load(pickle_2)
		comparison_pickle = pd.DataFrame(comparison_pickle)
		image_score = pd.DataFrame(image_score)

		f1 = comparison_pickle['f1'][:64]
		f2 = comparison_pickle['f2'][:64]
		image_id = comparison_pickle['imgId'][:64]
		f1_zip = list(zip(f1, image_id))
		f2_zip = list(zip(f2, image_id))
		# print("FACD_metadata loaded")
		comp_imgs_1 = [self.find_image(c, i) for c, i in f1_zip][:64]
		comp_imgs_2 = [self.find_image(c, i) for c, i in f2_zip][:64]
		# print("Comparison pairs and labels loaded")
		# (i,j, y): given items (i, j), y(i,j) = +1 indicates that i has a higher propensity to receive the absolute label yi = +1, 
		# compared to j. -1 otherwise.
		comp_labels = [1 if i=='left' else -1 for i in comparison_pickle['ans']][:64]
		abs_imgs = ['./FACD_image/Origin/' + file for file in os.listdir('./FACD_image/Origin')][:64]
		abs_image_class_map = {}
		for index, row in image_score.iterrows():
			if row['imgId'] not in abs_image_class_map:
				img_id = row['imgId']
				class_label = row['class']
				abs_image_class_map[img_id] = class_label
		abs_labels = [float(abs_image_class_map[file[:-4]]) for file in os.listdir('./FACD_image/Origin')][:64]

		# print("Absolute images and labels loaded ")
		# print("...converting to tensors")
		comp_imgs_1 = self.image_to_tensor(comp_imgs_1)
		# print("comp_imgs_1 converted to tensor")
		comp_imgs_2 = self.image_to_tensor(comp_imgs_2)
		# print("comp_imgs_2 converted to tensor")
		abs_imgs = self.image_to_tensor(abs_imgs)
		# print("abs_imgs converted to tensor")

		print("SHAPES OF DATA: ")
		print("abs_imgs: ", np.shape(abs_imgs))
		print("abs_labels: ", np.shape(abs_labels))
		print("comp_imgs_1: ", np.shape(comp_imgs_1))
		print("comp_imgs_2: ", np.shape(comp_imgs_2))
		print("comp_labels: ", np.shape(comp_labels))
		print("DONE LOADING DATA")
		return np.asarray(abs_imgs), np.asarray(abs_labels), np.asarray(comp_imgs_1), np.asarray(comp_imgs_2), np.asarray(comp_labels)

	def image_to_tensor(self, file_path_list, size = (3, 224, 224)):
		    output = []
		    for file_path in file_path_list:
		        im= np.asarray(Image.open(file_path).convert('RGB'))
		        im=cv2.resize(im, dsize=(size[1],size[2]))
		#         print(im.shape)
		        im = np.reshape(im, size)
		        output.append(im)
		    return output
	




