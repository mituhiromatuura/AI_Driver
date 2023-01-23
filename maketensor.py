'''
自動運転ラジコンカー
AIドライバーあいちゃん
http://ma2.la.coocan.jp/AI_Driver/

訓練前処理プログラム
maketensor.py
author mitshiro matsuura
version 1.0
date 2023.01.10
'''

import csv
import numpy as np
from PIL import Image

def make_tensor(img_w, img_h, datapath, filename):

	f = open(datapath + '/' + filename, 'r')
	csv_data = f.readlines()
	f.close()

	n = len(csv_data) - 1
	print('n = ', n)

	t1 = np.zeros((n, img_h, img_w, 3), np.uint8)
	t2 = np.array([], np.float32)
	t3 = np.array([], np.float32)

	f = open(datapath + '/' + filename, 'r')
	reader = csv.DictReader(f)

	n = 0
	for row in reader:

		jpg = row['jpg']
		servo = float(row['servo'])
		esc = float(row['esc'])
		print(jpg, servo, esc)

		cam_img = Image.open(datapath + '/' + jpg)
		cam_img = np.array(cam_img)
		cam_img = cam_img[:, :, ::-1]
		t1[n] = cam_img
		t2 = np.append(t2, servo)
		t3 = np.append(t3, esc)
		n += 1

	f.close()

	t1 = t1.astype('float32') / 255
	print(t1.nbytes)

	return t1, t2, t3
