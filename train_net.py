import os
import argparse
import numpy as np
import cv2
from pathlib import Path
import copy
import logging
import random
import scipy.io as sio
import time
# import matlab.engine

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.nn.functional as F
from torch.autograd import Variable

from srm_filter_kernel import all_normalized_hpf_list



# BOSSBASE_COVER_DIR = '/data/lml/jsgan/BossBase-1.01-cover-resample-256'
# BOWS_COVER_DIR = '/data/lml/jsgan/BOWS2-cover-resample-256'
COVER_DIR = '/data/lml/spa_test/BB-cover-resample-256'

IMAGE_SIZE = 256
BATCH_SIZE = 32 // 2
EPOCHS = 200
#EPOCHS = 2
LR = 0.01
WEIGHT_DECAY = 5e-4
EMBEDDING_RATE = 0.4

TRAIN_FILE_COUNT = 8000
TRAIN_PRINT_FREQUENCY = 100
EVAL_PRINT_FREQUENCY = 1
DECAY_EPOCH = [80, 140, 180]

#FINETUNE_EPOCHS = 1
FINETUNE_EPOCHS = 100


class TLU(nn.Module):
	def __init__(self, threshold):
		super(TLU, self).__init__()
		self.threshold = threshold

	
	def forward(self, input):
		output = torch.clamp(input, min=-self.threshold, max=self.threshold)

		return output





class HPF(nn.Module):
	def __init__(self):
		super(HPF, self).__init__()

		all_hpf_list_5x5 = []

		for hpf_item in all_normalized_hpf_list:
			if hpf_item.shape[0] == 3:
				hpf_item = np.pad(hpf_item, pad_width=((1, 1), (1, 1)), mode='constant')

			all_hpf_list_5x5.append(hpf_item)

		hpf_weight = nn.Parameter(torch.Tensor(all_hpf_list_5x5).view(30, 1, 5, 5), requires_grad=False)

		self.hpf = nn.Conv2d(1, 30, kernel_size=5, padding=2, bias=False)
		self.hpf.weight = hpf_weight
		self.tlu = TLU(3.0)


	def forward(self, input):
		output = self.hpf(input)
		output = self.tlu(output)

		return output




class Net(nn.Module):
	def __init__(self):
		super(Net, self).__init__()

		self.group1 = HPF()

		self.group2 = nn.Sequential(
			nn.Conv2d(30, 32, kernel_size=3, padding=1),
			nn.BatchNorm2d(32),
			nn.ReLU(),

			nn.Conv2d(32, 32, kernel_size=3, padding=1),
			nn.BatchNorm2d(32),
			nn.ReLU(),

			nn.Conv2d(32, 32, kernel_size=3, padding=1),
			nn.BatchNorm2d(32),
			nn.ReLU(),

			nn.Conv2d(32, 32, kernel_size=3, padding=1),
			nn.BatchNorm2d(32),
			nn.ReLU(),

			nn.AvgPool2d(kernel_size=3, padding=1, stride=2)
		)

		self.group3 = nn.Sequential(
			nn.Conv2d(32, 32, kernel_size=3, padding=1),
			nn.BatchNorm2d(32),
			nn.ReLU(),

			nn.Conv2d(32, 64, kernel_size=3, padding=1),
			nn.BatchNorm2d(64),
			nn.ReLU(),

			nn.AvgPool2d(kernel_size=3, padding=1, stride=2)
		)

		self.group4 = nn.Sequential(
			nn.Conv2d(64, 64, kernel_size=3, padding=1),
			nn.BatchNorm2d(64),
			nn.ReLU(),

			nn.Conv2d(64, 128, kernel_size=3, padding=1),
			nn.BatchNorm2d(128),
			nn.ReLU(),

			nn.AvgPool2d(kernel_size=3, padding=1, stride=2)
		)

		self.group5 = nn.Sequential(
			nn.Conv2d(128, 128, kernel_size=3, padding=1),
			nn.BatchNorm2d(128),
			nn.ReLU(),

			nn.Conv2d(128, 256, kernel_size=3, padding=1),
			nn.BatchNorm2d(256),
			nn.ReLU(),

			nn.AvgPool2d(kernel_size=32, stride=1)
		)

		self.fc1 = nn.Linear(1 * 1 * 256, 2)

	def forward(self, input):
		output = input

		output = self.group1(output)
		output = self.group2(output)
		output = self.group3(output)
		output = self.group4(output)
		output = self.group5(output)

		output = output.view(output.size(0), -1)
		output = self.fc1(output)

		return output




class AverageMeter(object):

	def __init__(self):
		self.reset()

	def reset(self):
		self.val = 0
		self.avg = 0
		self.sum = 0
		self.count = 0

	def update(self, val, n=1):
		self.val = val
		self.sum += val * n
		self.count += n
		self.avg = self.sum / self.count




def initWeights(module):
	if type(module) == nn.Conv2d:
		if module.weight.requires_grad:
			nn.init.kaiming_normal_(module.weight.data, mode='fan_in', nonlinearity='relu')

	if type(module) == nn.Linear:
		nn.init.normal_(module.weight.data, mean=0, std=0.01)
		nn.init.constant_(module.bias.data, val=0)




class AugData():
	def __call__(self, sample):
		data, label = sample['data'], sample['label']

		rot = random.randint(0,3)
		data = np.rot90(data, rot, axes=[1, 2]).copy()

		if random.random() < 0.5:
			data = np.flip(data, axis=2).copy()

		new_sample = {'data': data, 'label': label}

		return new_sample




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

	def __init__(self, stego_dir, index_path, transform=None):
		self.index_list = np.load(index_path)
		self.transform = transform

		# self.bossbase_cover_path = BOSSBASE_COVER_DIR + '/{}.pgm'
		# self.bows_cover_path = BOWS_COVER_DIR + '/{}.pgm'
		self.cover_path = COVER_DIR + '/{}.pgm'
		self.all_stego_path = stego_dir + '/{}.pgm'

	def __len__(self):
		return self.index_list.shape[0]

	def __getitem__(self, idx):
		file_index = self.index_list[idx]

		# if file_index <= 10000:
		# 	cover_path = self.bossbase_cover_path.format(file_index)
		# else:
		# 	cover_path = self.bows_cover_path.format(file_index - 10000)
		cover_path = self.cover_path.format(file_index)
		stego_path = self.all_stego_path.format(file_index)

		cover_data = cv2.imread(cover_path, -1)
		stego_data = cv2.imread(stego_path, -1)

		data = np.stack([cover_data, stego_data])
		label = np.array([0, 1], dtype='int32')

		sample = {'data': data, 'label': label}

		if self.transform:
			sample = self.transform(sample)

		return sample




def setLogger(log_path, mode='a'):
	logger = logging.getLogger()
	logger.setLevel(logging.INFO)

	if not logger.handlers:
		# Logging to a file
		file_handler = logging.FileHandler(log_path, mode=mode)
		file_handler.setFormatter(logging.Formatter('%(asctime)s: %(message)s', '%Y-%m-%d %H:%M:%S'))
		logger.addHandler(file_handler)

    	# Logging to console
		stream_handler = logging.StreamHandler()
		stream_handler.setFormatter(logging.Formatter('%(message)s'))
		logger.addHandler(stream_handler)





def train(model, device, train_loader, optimizer, epoch):

	batch_time = AverageMeter()
	data_time = AverageMeter()
	losses = AverageMeter()

	model.train()

	end = time.time()

	for i, sample in enumerate(train_loader):

		data_time.update(time.time() - end)

		data, label = sample['data'], sample['label']

		shape = list(data.size())
		data = data.reshape(shape[0] * shape[1], *shape[2:])
		label = label.reshape(-1)

		data, label = data.to(device), label.to(device)
		optimizer.zero_grad()
		output = model(data)

		criterion = nn.CrossEntropyLoss()
		loss = criterion(output, label)

		losses.update(loss.item(), data.size(0))

		loss.backward()
		optimizer.step()

		batch_time.update(time.time() - end)
		end = time.time()

		if i % TRAIN_PRINT_FREQUENCY == 0:
			logging.info('Epoch: [{0}][{1}/{2}]\t'
				'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
				'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
				'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
					epoch, i, len(train_loader), batch_time=batch_time,
					data_time=data_time, loss=losses))




def adjust_bn_stats(model, device, train_loader):

	model.train()

	with torch.no_grad():
		for sample in train_loader:
			data, label = sample['data'], sample['label']

			shape = list(data.size())
			data = data.reshape(shape[0] * shape[1], *shape[2:])
			label = label.reshape(-1)

			data, label = data.to(device), label.to(device)

			output = model(data)




def evaluate(model, device, eval_loader, epoch, optimizer, pt_path):

	model.eval()

	test_loss = 0
	correct = 0

	with torch.no_grad():
		for sample in eval_loader:
			data, label = sample['data'], sample['label']

			shape = list(data.size())
			data = data.reshape(shape[0] * shape[1], *shape[2:])
			label = label.reshape(-1)
			data, label = data.to(device), label.to(device)
			
			output = model(data)
			pred = output.max(1, keepdim=True)[1]
			correct += pred.eq(label.view_as(pred)).sum().item()

	accuracy = correct / (len(eval_loader.dataset) * 2)

	logging.info('-' * 8)
	logging.info('Eval accuracy: {:.4f}'.format(accuracy))

	all_state = {
		'original_state': model.state_dict(),
		'optimizer_state': optimizer.state_dict(),
		'epoch': epoch
	}

	torch.save(all_state, pt_path)
	logging.info('-' * 8)

	return accuracy













def trainNet(output_dir, stego_dir, gpu_num, train_index_path, val_index_path, test_index_path, fiter):

	log_path = output_dir + '/' + 'model-log'
	pt_path = output_dir + '/' + 'params.pt'
	print("\tsaved log path:", log_path)
	print("\tsaved checkpoint path:", pt_path)

	setLogger(log_path, mode='w')

	os.environ['CUDA_VISIBLE_DEVICES'] = gpu_num
	device = torch.device("cuda")
	kwargs = {'num_workers': 3, 'pin_memory': True}


	# load data
	train_transform = transforms.Compose([
		AugData(),
		ToTensor()
	])

	eval_transform = transforms.Compose([
		ToTensor()
	])

	train_dataset = MyDataset(stego_dir, index_path=train_index_path, transform=train_transform)
	valid_dataset = MyDataset(stego_dir, index_path=val_index_path, transform=eval_transform)
	test_dataset = MyDataset(stego_dir, index_path=test_index_path, transform=eval_transform)

	train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False, **kwargs)
	valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False, **kwargs)
	test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, **kwargs)


	# load model and params
	model = Net().to(device)
	model.apply(initWeights)

	params = model.parameters()

	params_wd, params_rest = [], []
	for param_item in params:
		if param_item.requires_grad:
			(params_wd if param_item.dim() != 1 else params_rest).append(param_item)

	param_groups = [{'params': params_wd, 'weight_decay': WEIGHT_DECAY},
					{'params': params_rest}]

	optimizer = optim.SGD(param_groups, lr=LR, momentum=0.9)


	# # load statepath
	# if (iteration == 0):
	# 	statePath = False
	# else:
	# 	statePath = True
	# 	statePath = output_dir + '/' + str(iteration-1) + '-' + 'params.pt'
	# print("\tstatePath:", statePath)

	# if statePath:
	# 	logging.info('-' * 8)
	# 	logging.info('Load state_dict in {}'.format(statePath))
	# 	logging.info('-' * 8)

	# 	all_state = torch.load(statePath)
	# 	original_state = all_state['original_state']
	# 	optimizer_state = all_state['optimizer_state']
	# 	epoch = all_state['epoch']

	# 	model.load_state_dict(original_state)
	# 	optimizer.load_state_dict(optimizer_state)

	# 	startEpoch = epoch + 1

	# else:
	# 	startEpoch = 1
	startEpoch = 1


	#if iteration == 15:
	#	FINETUNE_EPOCHS = 200


	# train & eval step
	scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=DECAY_EPOCH, gamma=0.1)


	if (fiter == True):

		for epoch in range(startEpoch, EPOCHS + 1):
			scheduler.step()

			train(model, device, train_loader, optimizer, epoch)

			if epoch % EVAL_PRINT_FREQUENCY == 0:
				adjust_bn_stats(model, device, train_loader)
				acc = evaluate(model, device, valid_loader, epoch, optimizer, pt_path)

		logging.info('\nTest set accuracy: \n')

		adjust_bn_stats(model, device, train_loader)
		acc = evaluate(model, device, valid_loader, epoch, optimizer, pt_path)

	else:

		for epoch in range(startEpoch, startEpoch+FINETUNE_EPOCHS):
			scheduler.step()

			train(model, device, train_loader, optimizer, epoch)

			if epoch % EVAL_PRINT_FREQUENCY == 0:
				adjust_bn_stats(model, device, train_loader)
				acc = evaluate(model, device, valid_loader, epoch, optimizer, pt_path)

		logging.info('\nTest set accuracy: \n')

		adjust_bn_stats(model, device, train_loader)
		acc = evaluate(model, device, valid_loader, epoch, optimizer, pt_path)


	return pt_path, acc

	
