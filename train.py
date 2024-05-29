'''
自動運転ラジコンカー
AIドライバーあいちゃん
http://ma2.la.coocan.jp/AI_Driver/

訓練プログラム
train.py
author mitsuhiro matsuura
version 1.0
date 2023.01.10
'''

import sys
from tensorflow.keras import layers, models, Input
import tensorflow as tf
from distutils.version import StrictVersion
from maketensor import make_tensor

if __name__ == "__main__":

	img_w = 160
	img_h = 80

	filename = 'log.csv'
	datapath = sys.argv[1]
	fit_epochs = int(sys.argv[2])

	if len(sys.argv) == 4:
		filename = sys.argv[3]

	t1, t2, t3 = make_tensor(img_w, img_h, datapath, filename)
	print(t1.shape, t2.shape, t3.shape)

	input_tensor = Input(shape=(img_h, img_w, 3))
	x = layers.Conv2D(16, 5, 2, activation = 'relu')(input_tensor)
	x = layers.Conv2D(32, 5, 2, activation = 'relu')(x)
	x = layers.Conv2D(64, 5, 2, activation = 'relu')(x)
	x = layers.Conv2D(128, 3, 1, activation = 'relu')(x)
	x = layers.Flatten()(x)
	x = layers.Dense(64, activation = 'relu')(x)

	print('Tensorflow version:', tf.__version__)
	if StrictVersion(tf.__version__) < StrictVersion('2.7.0'):
		output1_tensor = layers.Dense(1, name = 'servo')(x)
		output2_tensor = layers.Dense(1, name = 'esc')(x)
	else:
		output1_tensor = layers.Dense(1, name = 'esc')(x)
		output2_tensor = layers.Dense(1, name = 'servo')(x)

	model = models.Model(input_tensor, [output1_tensor, output2_tensor])

	print(model.summary())

	model.compile(
		loss = 'mse',
		optimizer = 'rmsprop',
		metrics = ['mae'])

	history = model.fit(t1, [t2, t3], epochs = fit_epochs)
	model.save('./model.h5')

	converter = tf.lite.TFLiteConverter.from_keras_model(model)
	tflite_model = converter.convert()
	with open('./model.tflite', 'wb') as f:
		f.write(tflite_model)
