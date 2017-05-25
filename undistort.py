import cv2
import cPickle as pickle
from image_generator import VideoCaptureCon


addr = "192.168.1.163"
user = "admin"
pswd = "uni-ubi20150119"
url = "rtsp://%s:%s@%s/H264?ch=1&subtype=0" % (user, pswd, addr)


def main():
    map1, map2 = pickle.load(open("data/map.pkl", "rb"))
    for view in VideoCaptureCon(url):
        rview = cv2.remap(view, map1, map2, cv2.INTER_LINEAR)
        cv2.imshow("Image RView", rview)
        cv2.imshow("Image View", view)
        c = cv2.waitKey(1)
        if c == 27:
            break        
        

if __name__ == "__main__":
    main()