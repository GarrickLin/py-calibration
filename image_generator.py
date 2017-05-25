import cv2
import threading
import Queue
import time
import logging

logging.basicConfig(level=logging.DEBUG,
                    format='(%(threadName)-9s) %(message)s',)

BUF_SIZE = 1
q = Queue.Queue(BUF_SIZE)


class ProducerThread(threading.Thread):
    def __init__(self, group=None, target=None, name=None, device_addr=None, verbose=None):
        super(ProducerThread, self).__init__()
        self.target = target
        self.name = name
        self.device_addr = device_addr

    def run(self):
        #url = "rtsp://admin:abcd1234@192.168.1.64:554/"
        # url = 0
        cap = cv2.VideoCapture(self.device_addr)
        while 1:
            ret, frame = cap.read()
            time.sleep(0.01)                       
            #logging.debug('reading loop...')
            if ret:
                if not q.full():
                    q.put(frame)
                    #logging.debug('Putting  frame' + ' : ' + str(q.qsize()) + ' items in queue')
                    # time.sleep(rand_time)
            else:
                print "device is not ready or done..."
                time.sleep(1)
                
        return
    

def VideoCaptureCon(addr, realtime=True):
    #if 1:
    #if isinstance(addr, int) or addr[:4] == "rtsp":    
    if realtime:
        p = ProducerThread(name='producer', device_addr=addr)
        p.start()
        time.sleep(2)       
        while 1:
            if not q.empty():
                yield q.get()
    else:
        cap = cv2.VideoCapture(addr)
        while 1:
            ret, frame = cap.read()
            if ret:
                yield frame
            else:
                print "device is not ready..."
                time.sleep(1)                
            

def gen_images(imglist):
    for imgname in imglist:
        img = cv2.imread(imgname)
        yield img
    yield None
    raise StopIteration

    
