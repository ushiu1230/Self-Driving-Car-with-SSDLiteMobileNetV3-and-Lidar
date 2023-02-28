# Python program to illustrate the concept
# of threading
# importing the threading module
import threading
import time

def videostream():
   start = time.time()
	# function to print cube of given num
   print("this is thread 1")
   time.sleep(3)
   print((time.time()-start))


def detecting():
   start = time.time()
   # function to print square of given num
   print("this is thread 2")
   time.sleep(3)
   print((time.time()-start))


if __name__ =="__main__":
	# creating thread
	t1 = threading.Thread(target= videostream)
	t2 = threading.Thread(target= detecting)

	# starting thread 1
	t1.start()
	# starting thread 2
	t2.start()

	# wait until thread 1 is completely executed
	t1.join()
	# wait until thread 2 is completely executed
	t2.join()

	# both threads completely executed
	print("Done!")
