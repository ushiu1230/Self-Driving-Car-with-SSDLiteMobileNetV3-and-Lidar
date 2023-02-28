import threading

# global variable x
x = 0

def increment():
	"""
	function to increment global variable x
	"""
	global x
	x += 1

def thread_task(lock):
	global x
	"""
    task for thread
	calls increment function 100000 times.
    """
	lock.acquire()
	x += 20000
	print('t1',x)
	lock.release()
    
    
def thread_print(lock):
    global x 
    lock.acquire()
    x += 20000
    print('t2',x)
    lock.release()
    
    

def main_task():
	global x
	# setting global variable x as 0
	x = 0

	# creating a lock
	lock = threading.Lock()
	while (x < 50000):
		increment()
		# creating threads
		t1 = threading.Thread(target=thread_task, args=(lock,))
		t2 = threading.Thread(target=thread_print, args=(lock,))

		# start threads
		t1.start()
		t2.start()

		# wait until threads finish their job
		t1.join()
		t2.join()

if __name__ == "__main__":
	for i in range(10):
		main_task()
		print("Iteration {0}: x = {1}".format(i,x))
