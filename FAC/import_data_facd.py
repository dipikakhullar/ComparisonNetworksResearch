import pickle
import pandas as pd
import numpy as np
import os
import torch
import torchvision
import torchvision.transforms as transforms
from PIL import Image
import cv2
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

class ImportData(object):
	def __init__(self, input_shape, data_dir=None):
		self.input_shape = input_shape
		self.data_dir = data_dir

	def find_image(self, category, image_id):
	    for file in os.listdir("./FACD_image/"+ category):
	        if image_id in file:
	            return "./FACD_image/"+ category + "/" + str(image_id) + ".jpg"

	def image_to_tensor(self, file_path_list, size = (3, 224, 224)):
	    output = []
	    for file_path in file_path_list:
	        im= np.asarray(Image.open(file_path).convert('RGB'))
	        im=cv2.resize(im, dsize=(size[1],size[2]))
	        im = np.reshape(im, size)
	        output.append(im)
	    return output

 	# def one_hot_encode_labels(self, labels):
		# label_encoder = LabelEncoder()
		# integer_encoded = label_encoder.fit_transform(labels)
		# # print(integer_encoded)
		# # binary encode
		# onehot_encoder = OneHotEncoder(sparse=False)
		# integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
		# onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
 	# 	return onehot_encoded

	def generate_comparison_data(self, category_df):
	    comp1 = []
	    comp2 = []
	    comp_labels =[]
	    labels_map = {"left": 1, "right": 0}
	    for index, row in category_df.iterrows():
	        img_id = row['imgId']
	        comp1_filter = row['f1']
	        comp2_filter = row['f2']
	        fname1 = "./FACD_image/"+ comp1_filter + "/" + str(img_id) + ".jpg"
	        fname2 = "./FACD_image/"+ comp2_filter + "/" + str(img_id) + ".jpg"
	        if os.path.isfile(fname1) and os.path.isfile(fname2):
	            comp1.append(fname1)
	            comp2.append(fname2)
	            comp_labels.append(labels_map[row['ans']])
	    return comp1, comp2, comp_labels


	def generate_absolute_labels(self, comparison_pickle, image_score):
		category_0 = comparison_pickle[comparison_pickle['category']==0]
		winning_filters = []
		for index, row in category_0.iterrows():
		    if row['ans'] == "left":
		        winning_filters.append(row['f1'])
		    else:
		        winning_filters.append(row['f2'])
		image_id_score_map = {}
		imgid = list(image_score["imgId"])
		filters = list(image_score["filterName"])
		classes = list(image_score["class"])
		for i in range(0, len(imgid)):
		    img_i = imgid[i]
		    class_i = classes[i]
		    filter_i = filters[i]
		    image_id_score_map[(img_i,filter_i)] = int(class_i)
		    


		class_column = []
		count = 0
		for index, row in category_0.iterrows():
		    img = row['imgId']
		    winning_filter = winning_filters[count]
		    to_add = image_id_score_map[img, winning_filter]
		    count += 1
		    class_column.append(to_add)

		category_0["class"]=class_column

		len(category_0)
		abs_images = []
		abs_labels = []
		for index, row in category_0.iterrows():
		    img_id = row["imgId"]
		    if row['ans'] == "left":
		        f1 = row['f1']
		        fname1 = "./FACD_image/"+ f1 + "/" + str(img_id) + ".jpg"
		        if os.path.isfile(fname1):
		            abs_images.append(fname1)
		            abs_labels.append(row['class'])
		    else:
		        f2 = row['f2']
		        fname2 = "./FACD_image/"+ f2 + "/" + str(img_id) + ".jpg"
		        if os.path.isfile(fname2):
		            abs_images.append(fname2)
		            abs_labels.append(row['class'])


		return abs_images, abs_labels 

	def load_data(self, datasplit=None):
		pickle_1 = open("FACD_metadata/pairwise_comparison.pkl","rb")
		pickle_2 = open("FACD_metadata/image_score.pkl","rb")
		comparison_pickle = pickle.load(pickle_1)
		image_score = pickle.load(pickle_2)
		comparison_pickle = pd.DataFrame(comparison_pickle)
		image_score = pd.DataFrame(image_score)

		category_0 = comparison_pickle[comparison_pickle['category']==0]
		category_0 = category_0[category_0['ans'].isin(["left", 'right'])]
		comp_imgs_1, comp_imgs_2, comp_labels = self.generate_comparison_data(category_0)
		abs_images, abs_labels = self.generate_absolute_labels(comparison_pickle, image_score)

		#one hot encoding values:
		label_encoder = LabelEncoder()
		integer_encoded = label_encoder.fit_transform(abs_labels)
		# print(integer_encoded)
		# binary encode
		onehot_encoder = OneHotEncoder(sparse=False)
		integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
		onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
		abs_labels = onehot_encoded
		print("abs_labels: ", np.shape(abs_labels))

		print("...converting to tensors")
		comp_imgs_1 = self.image_to_tensor(comp_imgs_1)
		# print("comp_imgs_1 converted to tensor")
		comp_imgs_2 = self.image_to_tensor(comp_imgs_2)
		# print("comp_imgs_2 converted to tensor")
		abs_imgs = self.image_to_tensor(abs_images)
		print("SHAPES OF DATA: ")
		print("abs_imgs: ", np.shape(abs_imgs))
		print("abs_labels: ", np.shape(abs_labels))
		print("comp_imgs_1: ", np.shape(comp_imgs_1))
		print("comp_imgs_2: ", np.shape(comp_imgs_2))
		print("comp_labels: ", np.shape(comp_labels))
		print("comparison labels: ", comp_labels)

		print("DONE LOADING DATA")
		return np.asarray(abs_imgs), np.asarray(abs_labels), np.asarray(comp_imgs_1), np.asarray(comp_imgs_2), np.asarray(comp_labels)





