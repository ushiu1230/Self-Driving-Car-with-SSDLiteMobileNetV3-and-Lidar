# This code will run the drone side, it will send video to cache server
# Lets import the libraries
# Welcome to PyShine
# www.pyshine.com
import socket, cv2, pickle, struct
import imutils
import cv2
import numpy as np

coco_names = [
    '__background__', 'Vehicle', 'Pedestrian'
]

server_socket = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
host_name  = socket.gethostname()
host_ip = socket.gethostbyname(host_name) # Enter the SERVER IP address
print('HOST IP:',host_ip)
port = 9999
socket_address = (host_ip,port)
server_socket.bind(socket_address)
server_socket.listen()
print("Listening at",socket_address)

def draw_boxes(boxes, classes, image):
    """
    Draws the bounding box around a detected object.
    """
    COLORS = (255, 0, 0)
    image = cv2.cvtColor(np.asarray(image), cv2.COLOR_BGR2RGB)
    for i, box in enumerate(boxes):
        cv2.rectangle(
            image,
            (int(box[0]), int(box[1])),
            (int(box[2]), int(box[3])),
            COLORS, 2
        )
        cv2.putText(image, classes[i], (int(box[0]), int(box[1]-5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, COLORS, 2, 
                    lineType=cv2.LINE_AA)
    return image


def start_video_stream():
	client_socket,addr = server_socket.accept()
	camera = False

	data = b""
	payload_size = struct.calcsize("Q")

	# if camera == True:
	# 	vid = cv2.VideoCapture(0)
	# else:
	vid = cv2.VideoCapture('D:/UIT/Study/KLTN/Object/Socket/Pov_Drive_to_Sunrise.mp4')
	print("isOpened: ",vid.isOpened())
	print('CLIENT {} CONNECTED!'.format(addr))
	
	# img,frame = vid.read()

	if client_socket:
		while(vid.isOpened()):
			img,frame = vid.read()
			# print('ret: ', img)
			frame  = imutils.resize(frame,width=256, height = 256)

			a = pickle.dumps(frame)
			message = struct.pack("Q",len(a))+a
			client_socket.sendall(message)
			# cv2.imshow("TRANSMITTING TO JETSON",frame)

			while len(data) < payload_size:
				packet = client_socket.recv(4*1024)
				if not packet: break
				data+=packet

			packed_msg_size = data[:payload_size]
			
			data = data[payload_size:]
			msg_size = struct.unpack("Q",packed_msg_size)[0]

			while len(data) < msg_size:
				data += client_socket.recv(4*1024)
			frame_data = data[:msg_size]
			data  = data[msg_size:]

			print_data = pickle.loads(frame_data)

			image_show = draw_boxes(boxes = print_data[0], classes = print_data[1], image = frame)

			# print('fps: ', print_data[2])
			cv2.imshow("recive data: ",image_show)

			key = cv2.waitKey(1) & 0xFF
			if key ==ord('q'):
				client_socket.close()
				cv2.destroyAllWindows()
				break

if __name__ == "__main__":
	start_video_stream()
