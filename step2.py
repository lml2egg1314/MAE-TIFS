import os
import argparse
import time
import numpy as np
import scipy.io as sio
# import matlab.engine
import shutil

import torch
import torch.nn as nn
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.nn.functional as F
from torch.autograd import Variable

import train_net
# import get_gradient, get_gradient_srnet

# hyperparameters
# T = 1  

# IMAGE_SIZE = 256
# TRAIN_BATCH_SIZE = 16
# TEST_BATCH_SIZE = 16


def myParseArgs():
	parser = argparse.ArgumentParser()

	parser.add_argument(
		'-g',
		'--gpuNum',
		help='Determine which gpu to use',
		type=str,
		choices=['0', '1', '2', '3'],
		required=True
	)

	parser.add_argument(
		'-p',
		'--payLoad',
		help='Determine the payload to embed',
		type=str,
		required=True
	)

	parser.add_argument(
		'-s',
		'--steganography',
		help='Determine the steganographic method to use',
		type=str,
		choices=['hill','suni','mipod','cmd_hill','wow'],
		required=True
	)

	# parser.add_argument('-s','--step',type=str,required=True)

	parser.add_argument(
		'-ln',
		'--listnum',
		help='Determine the list num to run',
		type=str,
		default='1',
		
		# required=True
	)

	args = parser.parse_args()
	
	return args





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


args = myParseArgs()

gpu_num = args.gpuNum
payload = float(args.payLoad)
list_num = int(args.listnum)
steganography = args.steganography
# step = args.step


# list & path
ALL_LIST = './index_list/' + str(list_num) + '/all_list.npy'
STEP1_LIST = './index_list/' + str(list_num) + '/train_and_val_list.npy'
STEP2_LIST = './index_list/' + str(list_num) + '/test_list.npy'
STEP1_TRAIN_LIST = './index_list/' + str(list_num) + '/train_list.npy'
STEP1_VAL_LIST = './index_list/' + str(list_num) + '/val_list.npy'
STEP2_TRAIN_LIST = './index_list/' + str(list_num) + '/retrain_train_list.npy'
STEP2_TEST_LIST = './index_list/' + str(list_num) + '/retrain_test_list.npy'



base_dir = '/data/lml/spa_test'.format(steganography)

cover_dir = '{}/BB-cover-resample-256'.format(base_dir)

sp_dir ='{}_{}'.format(steganography, payload)

stego_dir = '{}/{}/stego'.format(base_dir, sp_dir)


output_dir = './new/{}/{}'.format(list_num, sp_dir)

os.makedirs(output_dir, exist_ok = True)


# train net and save ckpt
print("2 - Train net in iteration")
train_net_time = AverageMeter()
start_train_net = time.time()

pt_name, it0_accuracy = train_net.trainNet(output_dir, stego_dir, gpu_num, STEP1_TRAIN_LIST, STEP1_VAL_LIST, STEP2_LIST, fiter=True)

print("\tIteration 0 DengNet Accuracy:", it0_accuracy)
train_net_time.update(time.time()-start_train_net)
print("2 - Train net in iteration: {:d}, \n\taccuracy: {:.3f}, \n\ttime: {:.3f}s".format(0, it0_accuracy, train_net_time.val))
