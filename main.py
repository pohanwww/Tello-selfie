from time import sleep
import sys
import tellopy
import keyboard
import traceback
import av
import cv2
import numpy
import time
import _thread

cascPath = r"D:\User\Documents\Python\Tello_selfie\FaceDetect-master\haarcascade_frontalface_default.xml"

lower = numpy.array([])
upper = numpy.array([])

def handler(event, sender, data, **args):
	drone = sender
	# if event is drone.EVENT_FLIGHT_DATA:
	# 	print(data)

def control(drone, vertical_bias, horizon_bias, detected_face, auto_mode):
	print("auto_mode:",detected_face)
	if auto_mode == False:
		if (keyboard.is_pressed('n')):
			drone.takeoff()
			sleep(2)
		elif (keyboard.is_pressed('i')):
			drone.up(50)
			sleep(0.5)
			drone.up(0)
			sleep(0.5)
		elif (keyboard.is_pressed('k')):
			drone.down(50)
			sleep(0.5)
			drone.down(0)
			sleep(0.5)
		elif (keyboard.is_pressed('w')):
			drone.forward(50)
			sleep(0.2)
			drone.forward(0)
			sleep(0.5)
		elif (keyboard.is_pressed('s')):
			drone.backward(50)
			sleep(0.2)
			drone.backward(0)
			sleep(0.5)
		elif (keyboard.is_pressed('a')):
			drone.left(50)
			sleep(0.2)	
			drone.left(0)
			sleep(0.5)
		elif (keyboard.is_pressed('d')):
			drone.right(50)
			sleep(0.2)
			drone.right(0)
			sleep(0.5)
		elif (keyboard.is_pressed('q')):
			drone.counter_clockwise(50)
			sleep(0.5)
			drone.counter_clockwise(0)
			sleep(0.5)
		elif (keyboard.is_pressed('e')):
			drone.clockwise(50)
			sleep(0.5)	
			drone.clockwise(0)
			sleep(0.5)
		elif (keyboard.is_pressed('z')):
			drone.frontflip()
			sleep(1)
		elif (keyboard.is_pressed('m')):	
			drone.down(50)
			sleep(5)
			drone.land()
			sleep(5)
		
	elif auto_mode == True: #480,360
		if detected_face == True:			
			if vertical_bias > 30:
				drone.down(int(vertical_bias/5))
				sleep(0.5)	
				drone.down(0)
				sleep(0.5)
			elif vertical_bias < -30:
				drone.up(abs(int(vertical_bias/5)))
				sleep(0.5)	
				drone.up(0)
				sleep(0.5)
			if horizon_bias > 60:
				drone.clockwise(int(horizon_bias/10))
				sleep(0.5)	
				drone.clockwise(0)
				sleep(0.5)
			elif horizon_bias < -60:
				drone.counter_clockwise(abs(int(horizon_bias/10)))
				sleep(0.5)	
				drone.counter_clockwise(0)
				sleep(0.5)
		
def start():
	drone = tellopy.Tello()
	try:
		drone.subscribe(drone.EVENT_FLIGHT_DATA, handler)
		drone.connect()
		drone.wait_for_connection(60.0)

		faceCascade = cv2.CascadeClassifier(cascPath)
		faceCascade.load(cascPath)
		container = av.open(drone.get_video_stream())
		frame_skip = 300
		frame_count = 0

		face_x = 480
		face_y = 360
		face_w = 0
		face_h = 0
		vertical_bias = 0
		horizon_bias = 0
		detected_face = False
		auto_mode = False
		pic_stable = False
		pic_time = 0
		pic_count = 0
		for frame in container.decode(video=0):
			if 0 < frame_skip:
				frame_skip = frame_skip - 1
				continue
			if (keyboard.is_pressed('c')):
				auto_mode = True
				print("c")
			elif (keyboard.is_pressed('v')):
				auto_mode = False
				print('v')
			_thread.start_new_thread( control, (drone, vertical_bias, horizon_bias, detected_face, auto_mode))

			start_time = time.time()
			image = cv2.cvtColor(numpy.array(frame.to_image()), cv2.COLOR_RGB2BGR)
			frame_count += 1
			print(frame_count)
			if frame_count > 4:
				gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
				faces = faceCascade.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=5,minSize=(30, 30))
				print("face",faces)
				if len(faces) > 0:
					face_x = faces[0][0]
					face_y = faces[0][1]
					face_w = faces[0][2]
					face_h = faces[0][3]
					detected_face = True
					
					if auto_mode == True:
						x = face_x + face_w/2
						y = face_y + face_h/2
						horizon_bias_ = x - 480
						vertical_bias_ = y - 360
						print("horizon_bias",horizon_bias)
						print("vertical_bias",vertical_bias)

						if vertical_bias_ > 30 and vertical_bias_ < -30:
							vertical_bias = vertical_bias_
						else:
							vertical_bias = 0
						if horizon_bias_ > 60 and horizon_bias_ < -60:
							horizon_bias = horizon_bias_
						else:
							horizon_bias = 0
						if vertical_bias == 0 and horizon_bias == 0:
							if pic_stable == False:
								pic_stable = True
								pic_time = time.time()
							elif pic_stable == True:
								if time.time() - pic_time > 3:
									pic_path = 'pictures/' + str(time.time()) + '.jpg'
									cv2.imwrite(pic_path, image)
									pic_count += 1
									pic_stable = False
									if pic_count == 3:
										auto_mode = False
								else:
									continue
						else:
							pic_stable = False
				else:
					detected_face = False
				frame_count = 0
			if detected_face == True:
				cv2.rectangle(image, (face_x, face_y), (face_x+face_w, face_y+face_h), (0, 255, 0), 2)

			cv2.imshow("Faces found", image)
			#################################

			# # Create the haar cascade
			# faceCascade = cv2.CascadeClassifier(cascPath)

			# # Read the image
			# # image = cv2.imread(imagePath)
			# gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
			# # faceCascade.load(cascPath)
			# # Detect faces in the image
			# faces = faceCascade.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=5,minSize=(30, 30))
			# 	#flags = cv2.CV_HAAR_SCALE_IMAGE)

			# print("Found {0} faces!".format(len(faces)))

			# # Draw a rectangle around the faces
			# for (x, y, w, h) in faces:
			# 	cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
			
			####################################
			# cv2.imshow('Original', image)
			# print("sssss", time.time() - st)

			# frame_skip = int((time.time() - start_time)/frame.time_base)
			frame_skip = int((time.time() - start_time)/(1/100))

			if cv2.waitKey(1) == 27:
				break
	finally:
		drone.quit()
		cv2.destroyAllWindows()
if __name__ == '__main__':
	start()
	
	"""cnt = sorted(cnts, key = cv2.contourArea, reverse = True)[0]
                rect = np.int32(cv2.boxPoints(cv2.minAreaRect(cnt)))
                cv2.drawContours(image, [rect], -1, (0, 255, 0), 2)
                object_x = (rect[0][0] + rect[1][0] + rect[2][0] + rect[3][0]) / 4
                object_y = (rect[0][1] + rect[1][1] + rect[2][1] + rect[3][1]) / 4
                if object_x < 250: 
                    drone.counter_clockwise(40) 
                    print('left')
                elif object_x > 390:  
                    drone.clockwise(40)
                    print('right')  
                else :
                    drone.counter_clockwise(0)
                    drone.clockwise(0)"""