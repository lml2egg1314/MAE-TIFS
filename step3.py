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
import get_gradient_mm_mipod
import get_gradient_mm




# hyperparameters
T = 1

IMAGE_SIZE = 256
TRAIN_BATCH_SIZE = 16
TEST_BATCH_SIZE = 16


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
		help='Determine the payload to embed',
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

base_data_dir = '/data/lml/spa_test/{}'.format(sp_dir)

stego_dir = '{}/{}/stego'.format(base_dir, sp_dir)


output_dir = './{}/{}'.format(list_num, sp_dir)
dnet_pt_name = '{}/params.pt'.format(output_dir)



grad_dir = '{}/{}/{}/mm_grad_10'.format(base_dir, sp_dir, list_num)

prob_dir = '{}/{}/prob'.format(base_dir, sp_dir)
print("3 - Calculate and save gradient")
gen_grad_time = AverageMeter()
start_gen_grad = time.time()
if steganography == 'mipod':
	get_gradient_mm_mipod.calSignGrad(prob_dir, dnet_pt_name, ALL_LIST, grad_dir, gpu_num)
else:
	get_gradient_mm.calSignGrad(prob_dir, dnet_pt_name, ALL_LIST, grad_dir, gpu_num)
gen_grad_time.update(time.time()-start_gen_grad)
print("3 - Calculate and save gradient: {:.3f}s".format(gen_grad_time.val))


