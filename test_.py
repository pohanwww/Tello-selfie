import time
import sys
import tellopy
import keyboard
import traceback
import av
import cv2
import numpy
import time
# import _thread

# def handler(event, sender, data, **args):
# 	drone = sender
# 	if event is drone.EVENT_FLIGHT_DATA:
# 		print(data)

def mainloop():
	drone = tellopy.Tello()
	try:
		# drone.subscribe(drone.EVENT_FLIGHT_DATA, handler)
		drone.connect()
		drone.wait_for_connection(60.0)
		container = av.open(drone.get_video_stream())
		stream = container.streams.video[0]
		print('streea',stream)
		print(stream.time_base)
		frame_skip = 300
		for index, frame in enumerate(container.decode(stream)):
			print('index:', index)
			if 0 < frame_skip:
				frame_skip = frame_skip - 1
				continue
			# _thread.start_new_thread( control, (drone,))
			start_time = time.time()
			image = cv2.cvtColor(numpy.array(frame.to_image()), cv2.COLOR_RGB2BGR)
			print('hi1')
			cv2.imshow('Original', image)
			print('hi2')
	
			print('jijijijij',container.decode(video=0))
			frame_skip = int((time.time() - start_time)/(1/100))
			print('streea',stream)
			print(stream.time_base)
			print('hi3')
			print('video')
			if cv2.waitKey(1) == 27:
				print('break')
				break
	finally:
		print('stoped')
		drone.quit()
		cv2.destroyAllWindows()
if __name__ == '__main__':
	mainloop()