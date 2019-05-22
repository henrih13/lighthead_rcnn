import json
import numpy as np
import cv2
import glob
import argparse

path_picture = '/home/erik/Documents/light_head_rcnn-master/create_annotation/Uploadme/lastanno/*.png'
pictures = []
for filename in glob.glob(path_picture):
	pictures.append(filename) 

def create_data(args):
	path = '/home/erik/Documents/light_head_rcnn-master/create_annotation/*'+args+'*.json'
	path_template = '/home/erik/Documents/light_head_rcnn-master/create_annotation/template.json'


	for filename in glob.glob(path):
		with open(filename, 'r') as f:
			for line in f:
				file = line

	for filename in glob.glob(path_template):
		with open(filename, 'r') as f:
			for line in f:
				file_template = line
	data = json.loads(file)
	data_template = json.loads(file_template)
	data_tmp = json.loads(file_template)


	path_filec = '/home/erik/Documents/light_head_rcnn-master/create_annotation/results/mb_'+args+'.odgt'
	filec = open(path_filec, 'w+')
	for i in data:
		data_tmp['fpath'] = '/'+args+'/'+data[i]['filename']
		data_tmp['ID'] = data[i]['filename']
		pic_idx = 0
		for j in range(len(pictures)):
			if pictures[j].split('/')[-1] == data[i]['filename']:
				pic_idx = j
		img = cv2.imread(pictures[pic_idx])
		data_tmp['height'] = img.shape[0]
		data_tmp['width'] = img.shape[1]
		
		for j in range(len(data[i]['regions'])):
			data_tmp['gtboxes'].append({"box": [0,0,0,0], "occ": 0, "tag":"", "extra": {"ignore": 0}})
			tmp_list = []
			for k in data[i]['regions'][j]['shape_attributes']:
				if k != 'name':
					tmp_list.append(data[i]['regions'][j]['shape_attributes'][k])
			data_tmp['gtboxes'][j]['box'] = tmp_list
			tmp_list = []
			for k in data[i]['regions'][j]['region_attributes']:
				if data[i]['regions'][j]['region_attributes'][k]:
					data_tmp['gtboxes'][j]['tag'] = data[i]['regions'][j]['region_attributes'][k]
		

		filec.write(str(data_tmp)+'\n')
		data_tmp = json.loads(file_template)
	f.close

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Specify train/val')
	parser.add_argument('-t',type=str, default="train")
	args = parser.parse_args()
	create_data(args.t)

