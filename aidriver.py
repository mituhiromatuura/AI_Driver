'''
自動運転ラジコンカー
AIドライバーあいちゃん
http://ma2.la.coocan.jp/AI_Driver/

走行プログラム
aidriver.py
author mitsuhiro matsuura
version 1.01
date 2023.01.22
'''

import concurrent.futures
import csv
import cv2
import multiprocessing
import numpy as np
import os
import pigpio
import sys
import struct
import tensorflow as tf
import time

import cfg

port = pigpio.pi()

q_cam = multiprocessing.Queue()
q_img = multiprocessing.Queue(0)
v_auto_on = multiprocessing.Value('b', False)
v_esc = multiprocessing.Value('f', 0)
v_esc_on = multiprocessing.Value('b', False)
v_fps = multiprocessing.Value('f', 30)
v_log_on = multiprocessing.Value('b', False)
v_model = multiprocessing.Value('b', False)
v_num = multiprocessing.Value('I', 0)
v_ok = multiprocessing.Value('b', True)
v_pwm1 = multiprocessing.Value('I', 0)
v_pwm2 = multiprocessing.Value('I', 0)
v_servo = multiprocessing.Value('f', 0)
v_servo_trim = multiprocessing.Value('f', 0.0)
v_speed_trim = multiprocessing.Value('f', 1.0)
v_speed_trim_ai = multiprocessing.Value('f', 1.0)
v_tick1 = multiprocessing.Value('I', 0)
v_tick2 = multiprocessing.Value('I', 0)
v_trigger = multiprocessing.Value('f', 0)
v_wheel = multiprocessing.Value('f', 0)

def button(pin, led, ch):
	print('button(pin=' + str(pin) + ') start')

	port.set_pull_up_down(pin, pigpio.PUD_UP)
	port.set_mode(pin, pigpio.INPUT)

	port.set_mode(led, pigpio.OUTPUT)
	port.write(led, not ch.value)

	sw_on = False
	while v_ok.value:
		if sw_on:
			if port.read(pin) == 1:
				sw_on = False
		else:
			if port.read(pin) == 0:
				sw_on = True
				ch.value = not ch.value
		port.write(led, not ch.value)
		time.sleep(0.1)

	port.write(led, 1)
	print('button(pin=' + str(pin) + ') end')

def hw_pwm(pin, ch, center, max, min):
	print('hw_pwm(pin=' + str(pin) + ') start')

	port.set_mode(pin, pigpio.OUTPUT)
	freq = 100

	max_range = max - center
	min_range = center - min

	while v_ok.value:
		#pulse = (ch.value * range + center) * 100
		if ch.value > 0:
			pulse = (ch.value * max_range + center) * 100
		else:
			pulse = (ch.value * min_range + center) * 100
		port.hardware_PWM(pin, freq, int(pulse))
		time.sleep(1.0 / v_fps.value)

	print('hw_pwm(pin=' + str(pin) + ') end')

def limit(value):
	value = min(value, 1.0)
	value = max(-1.0, value)
	return value

def callback(gpio, level, tick):
	if gpio == cfg.GPIO_WHEEL:
		if level == 1:
			v_tick1.value = tick
		else:
			v_pwm1.value = tick - v_tick1.value
			offset = tick - v_tick1.value - cfg.PWM_WHEEL_CENTER
			if offset > 0:
				v_wheel.value = limit(offset / (cfg.PWM_WHEEL_RIGHT - cfg.PWM_WHEEL_CENTER))
			else:
				v_wheel.value = limit(offset / (cfg.PWM_WHEEL_CENTER - cfg.PWM_WHEEL_LEFT))

	elif gpio == cfg.GPIO_TRIGGER:
		if level == 1:
			v_tick2.value = tick
		else:
			v_pwm2.value = tick - v_tick2.value
			offset = tick - v_tick2.value - cfg.PWM_TRIGGER_CENTER
			if offset > 0:
				v_trigger.value = limit(offset / (cfg.PWM_TRIGGER_FORWARD - cfg.PWM_TRIGGER_CENTER))
			else:
				v_trigger.value = limit(offset / (cfg.PWM_TRIGGER_CENTER - cfg.PWM_TRIGGER_REVERSE))

def pwm_get(pin):
	print('pwm_get(pin=' + str(pin) + ') start')

	port.set_mode(pin, pigpio.INPUT)
	cb = port.callback(pin, pigpio.EITHER_EDGE, callback)

	while v_ok.value:
		time.sleep(1)

	cb.cancel()
	print('pwm_get(pin=' + str(pin) + ') end')

def pwm_check():
	print('pwm_check start')

	while v_ok.value:
		p1 = 0
		p2 = 0
		for i in range(10):
			p1 += v_pwm1.value
			p2 += v_pwm2.value
			time.sleep(0.1)
		print('WHEEL PWM =', p1 // 10, '(usec) TRIGGER PWM =', p2 // 10, '(usec)')

	v_ok.value = False
	print('pwm_check end')

def camera(fps, width, height, rotate):
	print("camera start")

	capture0 = cv2.VideoCapture(0)

	capture0.set(cv2.CAP_PROP_FRAME_WIDTH, width)
	capture0.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
	capture0.set(cv2.CAP_PROP_FPS, fps)

	print('width:', capture0.get(cv2.CAP_PROP_FRAME_WIDTH), '(', width, ')')
	print('height:', capture0.get(cv2.CAP_PROP_FRAME_HEIGHT), '(', height, ')')
	print('fps:', capture0.get(cv2.CAP_PROP_FPS), '(', fps, ')')

	v_fps.value = fps

	q_cam.get()

	while v_ok.value:
		ret, frame = capture0.read()
		if rotate:
			frame = cv2.rotate(frame, cv2.ROTATE_180)
		q_img.put_nowait(frame)

	print('camera end')

def play(log_dir):
	print("play start")

	f = open(log_dir + '/log.csv','r')
	reader = csv.DictReader(f)

	q_cam.get()
	v_esc_on.value = True

	for row in reader:
		if not v_ok.value:
			break
		index = int(row['index'])
		print(index)
		time2 = float(row['time2'])
		cam_file = row['jpg']
		v_wheel.value = float(row['servo'])
		v_trigger.value = float(row['esc'])

		frame = cv2.imread(log_dir + '/' + cam_file, cv2.IMREAD_UNCHANGED)
		q_img.put(frame)
		time.sleep(time2)

	f.close()
	print('play end')
	v_ok.value = False
	q_img.put(frame)

class Fpv:
	def open(self):
		self.BLACK   = (  0,  0,  0)
		self.RED     = (0,  0,  255)
		self.GREEN   = (0,  255,0  )
		self.YELLOW  = (0,  255,255)
		self.BLUE    = (255,0,  0  )
		self.MAGENTA = (255,  0,255)
		self.CYAN    = (255,255,  0)
		self.WHITE   = (255,255,255)

		window_style = cv2.WINDOW_NORMAL | cv2.WINDOW_GUI_NORMAL
		cv2.namedWindow('Camera', window_style)

	def disp(self, frame, servo, esc):
		if not v_esc_on.value:
			color = self.WHITE
		elif v_auto_on.value and v_log_on.value:
			color = self.MAGENTA
		elif v_auto_on.value:
			color = self.YELLOW
		elif v_log_on.value:
			color = self.RED
		else:
			color = self.GREEN

		height, width, _ = frame.shape
		x0=int(round(width/2))
		y0=int(round(height))
		x1=int(round(width/2 + width * servo / 2))
		y1=int(round(height - height * esc))
		cv2.line(frame,(x0 ,y0),(x1,y1),color,2)
		cv2.imshow('Camera', frame)

		wk = cv2.waitKey(1) & 0xFF
		if wk == ord('q'):
			return False
		elif wk == ord('w'):
			v_esc_on.value = not v_esc_on.value
			print('ESC', 'ON' if v_esc_on.value else 'OFF')
		elif wk == ord('e'):
			v_auto_on.value = not v_auto_on.value
			print('AUTO', 'ON' if v_auto_on.value else 'OFF')
		elif wk == ord('r'):
			v_log_on.value = not v_log_on.value
			print('LOG', 'ON' if v_log_on.value else 'OFF')
		return True

	def close(self):
		cv2.destroyAllWindows()

class Log:
	def __init__(self, log_dir):
		self.log_dir = log_dir

		if not os.path.isdir(self.log_dir):
			os.makedirs(self.log_dir)
		if not os.path.isdir(self.log_dir + 'jpg'):
			os.makedirs(self.log_dir + 'jpg')

		self.n = 0
		self.log_index = list()
		self.log_time1 = list()
		self.log_time2 = list()
		self.log_time3 = list()
		self.log_jpg = list()
		self.log_servo = list()
		self.log_esc = list()

	def append(self, t1, t2, t3, frame, servo, esc):
		self.n += 1
		jpg_file_name = 'jpg/cam_{:05d}.jpg'.format(self.n)
		cv2.imwrite(self.log_dir + jpg_file_name, frame, [cv2.IMWRITE_JPEG_QUALITY, 50])

		self.log_index.append(self.n)
		self.log_time1.append(t1)
		self.log_time2.append(t2)
		self.log_time3.append(t3)
		self.log_jpg.append(jpg_file_name)
		self.log_servo.append(servo)
		self.log_esc.append(esc)

		if self.n % 100 == 0:
			print('log count:', self.n)
		return self.n

	def close(self):
		if self.n > 0:
			f = open(self.log_dir + 'log.csv', 'w')
			f.write( \
				'index,' + \
				'time1,' + \
				'time2,' + \
				'time3,' + \
				'jpg,' + \
				'servo,' + \
				'esc,' + \
				'\n')
			for i in range(self.n):
				f.write(str(self.log_index[i]) + ',')
				f.write(str(self.log_time1[i]) + ',')
				f.write(str(self.log_time2[i]) + ',')
				f.write(str(self.log_time3[i]) + ',')
				f.write(self.log_jpg[i] + ',')
				f.write(str(self.log_servo[i]) + ',')
				f.write(str(self.log_esc[i]) + ',')
				f.write('\n')
			f.close()
			print('log.csv saved', self.n)

class Model:
	def __init__(self, model_path):
		self.model_path = model_path
		if os.path.exists(model_path):
			print('model file', model_path, 'loading')
			self.interpreter = tf.lite.Interpreter(model_path)
			self.interpreter.allocate_tensors()
			self.input_details = self.interpreter.get_input_details()
			self.output_details = self.interpreter.get_output_details()
			print('model load OK')
			v_model.value = True
		else:
			print('model file', model_path, 'is not exists')
			v_model.value = False

	def predict(self, frame):
		if v_model.value:
			x = frame
			x = x.astype('float32') / 255
			x = np.expand_dims(x, axis=0)

			self.interpreter.set_tensor(self.input_details[0]['index'], x)
			self.interpreter.invoke()

			output_data0 = self.interpreter.get_tensor(self.output_details[0]['index'])
			output_data1 = self.interpreter.get_tensor(self.output_details[1]['index'])

			servo = output_data0[0][0]
			esc = output_data1[0][0]
			return servo, esc
		else:
			if v_auto_on.value:
				print('AUTO OFF', self.model_path, 'is none')
				v_auto_on.value = False
			return 0, 0

class Speed:
	def __init__(self, speed_trim, speed_trim_ai):
		v_speed_trim.value = speed_trim
		v_speed_trim_ai.value = speed_trim_ai

	def trim(self, esc):
		esc *= v_speed_trim.value

		if v_auto_on.value:
			if 0.9 < v_trigger.value:
				#手動停止
				esc = -1.0
			else:
				esc *= v_speed_trim_ai.value
		return esc

def drive(fpv, log, model, speed):
	print("drive start")

	fpv.open()
	q_cam.put(True)
	t1 = time.time()

	while v_ok.value:
		t2 = time.time() - t1
		t1 = time.time()

		frame = q_img.get()

		t3 = time.time()
		if v_auto_on.value:
			servo, esc = model.predict(frame)
		else:
			servo = v_wheel.value
			esc = v_trigger.value
		t3 = time.time() - t3

		v_servo.value = limit(servo + v_servo_trim.value)
		if v_esc_on.value:
			v_esc.value = limit(speed.trim(esc))
		else:
			v_esc.value = 0

		if v_log_on.value:
			v_num.value = log.append(t1, t2, t3, frame, servo, esc)

		if not fpv.disp(frame, servo, esc):
			v_ok.value = False

	log.close()
	fpv.close()
	v_ok.value = False
	print('drive end')

if __name__ == "__main__":
	print("main start")

	fpv = Fpv()
	log = Log(cfg.LOG_DIR)
	model = Model(cfg.MODEL_FILE)
	speed = Speed(cfg.SPEED_TRIM, cfg.SPEED_TRIM_AI)

	executor_t = concurrent.futures.ThreadPoolExecutor(max_workers=13)

	mode = 'drive'
	if len(sys.argv) == 2:
		mode = sys.argv[1]

	if mode == 'pwm':
		executor_t.submit(pwm_get, cfg.GPIO_WHEEL)
		time.sleep(0.1)
		executor_t.submit(pwm_get, cfg.GPIO_TRIGGER)
		executor_t.submit(pwm_check)

	elif mode == 'play':
		executor_t.submit(play, cfg.LOG_DIR)
		executor_t.submit(drive, fpv, log, model, speed)
		executor_t.submit(button, cfg.GPIO_BTN_B, cfg.GPIO_LED_B, v_auto_on)

	elif mode == 'drive':
		executor_t.submit(camera, cfg.CAMERA_FPS, cfg.FRAME_WIDTH, cfg.FRAME_HEIGHT, cfg.CAM_ROTATE)
		executor_t.submit(drive, fpv, log, model, speed)
		executor_t.submit(button, cfg.GPIO_BTN_A, cfg.GPIO_LED_A, v_esc_on)
		executor_t.submit(button, cfg.GPIO_BTN_B, cfg.GPIO_LED_B, v_auto_on)
		executor_t.submit(button, cfg.GPIO_BTN_C, cfg.GPIO_LED_C, v_log_on)
		executor_t.submit(hw_pwm, cfg.GPIO_SERVO, v_servo, cfg.PWM_WHEEL_CENTER, cfg.PWM_WHEEL_RIGHT, cfg.PWM_WHEEL_LEFT)
		executor_t.submit(hw_pwm, cfg.GPIO_ESC, v_esc, cfg.PWM_TRIGGER_CENTER, cfg.PWM_TRIGGER_FORWARD, cfg.PWM_TRIGGER_REVERSE)
		executor_t.submit(pwm_get, cfg.GPIO_WHEEL)
		time.sleep(0.1)
		executor_t.submit(pwm_get, cfg.GPIO_TRIGGER)

	print('main end')
