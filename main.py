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

lower = numpy.array([])
upper = numpy.array([])

def handler(event, sender, data, **args):
	drone = sender
	if event is drone.EVENT_FLIGHT_DATA:
		print(data)
def control(drone):
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
		
def test():
	drone = tellopy.Tello()
	try:
		drone.subscribe(drone.EVENT_FLIGHT_DATA, handler)
		drone.connect()
		drone.wait_for_connection(60.0)
		container = av.open(drone.get_video_stream())
		frame_skip = 300
		for frame in container.decode(video=0):
			if 0 < frame_skip:
				frame_skip = frame_skip - 1
				continue
			_thread.start_new_thread( control, (drone,))
			start_time = time.time()
			image = cv2.cvtColor(numpy.array(frame.to_image()), cv2.COLOR_RGB2BGR)
			cv2.imshow('Original', image)
			# frame_skip = int((time.time() - start_time)/frame.time_base)
			frame_skip = int((time.time() - start_time)/(1/100))
			print('video')
			if cv2.waitKey(1) == 27:
				break
	finally:
		drone.quit()
		cv2.destroyAllWindows()
if __name__ == '__main__':
	test()
	
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