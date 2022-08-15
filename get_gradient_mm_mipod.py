
import os
import argparse
import numpy as np
import cv2
import logging
import random
import shutil
import time
import math
from PIL import *
import PIL.Image
import scipy.io as sio
from pathlib import Path
# import matlab.engine

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.nn.functional as F
from torch.autograd import Variable

import train_net



# BOSSBASE_COVER_DIR = '/data/lml/jsgan/BossBase-1.01-cover-resample-256'
# BOWS_COVER_DIR = '/data/lml/jsgan/BOWS2-cover-resample-256'
BB_COVER_DIR = '/data/lml/spa_test/BB-cover-resample-256'
BATCH_SIZE = 1


class ToTensor():
	def __call__(self, sample):
		data, label = sample['data'], sample['label']

		data = np.expand_dims(data, axis=1)
		data = data.astype(np.float32)

		new_sample = {
			'data': torch.from_numpy(data),
			'label': torch.from_numpy(label).long(),
		}

		return new_sample




class MyDataset(Dataset):
	def __init__(self, prob_dir, index_path, transform=None):
		self.index_list = np.load(index_path)
		self.transform = transform
		# self.stego_numbers = 5
		self.cover_path = BB_COVER_DIR + '/{}.pgm'
		self.prob_path = prob_dir + '/{}.mat'

	def __len__(self):
		return self.index_list.shape[0]

	def __getitem__(self, idx):
		file_index = self.index_list[idx]

		cover_path = self.cover_path.format(file_index)

		cover_data = cv2.imread(cover_path, -1)

		prob_mat = sio.loadmat(self.prob_path.format(file_index))
		prob = prob_mat['prob_map']
		pp1 = prob/2
		pm1 = prob/2

		n = np.random.rand(10,256,256)
		m = np.zeros([10,256,256])

		m[pp1 > n] = 1
		m[pm1 > 1-n] = -1

		stego_data = cover_data + m

		cover_data = np.expand_dims(cover_data, axis=0)
		cover_data = cover_data.astype(np.float32)

		data = np.concatenate([cover_data, stego_data], axis=0)
		label = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype='int32')

		sample = {'data': data, 'label': label}

		if self.transform:
			sample = self.transform(sample)

		return sample



def calSignGrad(prob_dir, pt_path, indexPath, grad_dir, gpu_num):

	print("\tread checkpoint path:", pt_path)
	print("\tsaved grad path:", grad_dir)

	Path(grad_dir).mkdir(parents=True, exist_ok=True)


	os.environ['CUDA_VISIBLE_DEVICES'] = gpu_num
	device = torch.device("cuda")
	kwargs = {'num_workers': 1, 'pin_memory': True}

	data_transform = transforms.Compose([
		ToTensor()
	])
	data_dataset = MyDataset(prob_dir, index_path=indexPath, transform=data_transform)
	data_loader = DataLoader(data_dataset, batch_size=BATCH_SIZE, shuffle=False, **kwargs)
	index_list = np.load(indexPath)

	model = train_net.Net().to(device)
	all_state = torch.load(pt_path)
	model.load_state_dict(all_state['original_state'])
	model.eval()

	#torch.set_printoptions(edgeitems=5)
	
	for i, sample in enumerate(data_loader):

		file_index = index_list[i]
		#print(str(i+1), "-", file_index, "---------------------")

		data, label = sample['data'], sample['label']
		shape = list(data.size())
		data = data.reshape(shape[0] * shape[1], *shape[2:])
		label = label.reshape(-1)

		data, label = data.to(device), label.to(device)
		data.requires_grad = True

		output = model(data)
		criterion = nn.CrossEntropyLoss()
		loss = criterion(output, label)

		model.zero_grad()
		loss.backward()

		grad = data.grad.data
		mm_grad = grad.cpu().numpy().squeeze()
		# sign_grad = torch.sign(grad)
		# sign_grad = sign_grad.cpu().numpy().squeeze()
		# temp_grad = grad.cpu().numpy().squeeze()


		#abs_bool = calAbsBool(threshold, grad)
		
		sio.savemat('{}/{}.mat'.format(grad_dir, str(file_index)), mdict={'mm_grad':mm_grad})
		

		# sio.savemat('{}/{}.mat'.format(grad_dir, str(file_index)), mdict={'sign_grad':sign_grad, 'grad':temp_grad})


def caloutput(data_dir, pt_path, indexPath, grad_dir, gpu_num):

	print("\tread checkpoint path:", pt_path)
	print("\tsaved grad path:", grad_dir)

	# Path(grad_dir).mkdir(parents=True, exist_ok=True)


	os.environ['CUDA_VISIBLE_DEVICES'] = gpu_num
	device = torch.device("cuda")
	kwargs = {'num_workers': 1, 'pin_memory': True}

	data_transform = transforms.Compose([
		ToTensor()
	])
	data_dataset = MyDataset(data_dir, index_path=indexPath, transform=data_transform)
	data_loader = DataLoader(data_dataset, batch_size=BATCH_SIZE, shuffle=False, **kwargs)
	index_list = np.load(indexPath)

	model = train_net.Net().to(device)
	all_state = torch.load(pt_path)
	model.load_state_dict(all_state['original_state'])
	model.eval()

	
	#torch.set_printoptions(edgeitems=5)
	all_score = torch.zeros(len(index_list),2).numpy()
	for i, sample in enumerate(data_loader):

		file_index = index_list[i]
		#print(str(i+1), "-", file_index, "---------------------")

		data, label = sample['data'], sample['label']
		shape = list(data.size())
		data = data.reshape(shape[0] * shape[1], *shape[2:])
		label = label.reshape(-1)

		data, label = data.to(device), label.to(device)
		# data.requires_grad = True

		output = model(data)
		output = output.detach().cpu().numpy().squeeze()

		all_score[file_index - 1] = output
		# criterion = nn.CrossEntropyLoss()
		# loss = criterion(output, label)

		# model.zero_grad()
		# loss.backward()

		# grad = data.grad.data
		# sign_grad = torch.sign(grad)
		# sign_grad = sign_grad.cpu().numpy().squeeze()
		# temp_grad = grad.cpu().numpy().squeeze()


		#abs_bool = calAbsBool(threshold, grad)
		
		

	sio.savemat(grad_dir, mdict={'all_score':all_score})




	