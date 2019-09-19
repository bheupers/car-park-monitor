# USAGE
# python client.py --server-ip SERVER_IP

# import the necessary packages
from imutils.video import VideoStream
from threading import Thread
import cv2
import imagezmq
import argparse
import signal
import socket
import sys
import time


def gstreamer_pipeline(capture_width=1280, capture_height=720, display_width=1280, display_height=720, framerate=60,
                       flip_method=0):
    """
    gstreamer_pipeline returns a GStreamer pipeline for capturing from the CSI camera
    Defaults to 1280x720 @ 60fps
    Flip the image by setting the flip_method (most common values: 0 and 2)
    display_width and display_height determine the size of the window on the screen
    """
    return ('nvarguscamerasrc ! '
            'video/x-raw(memory:NVMM), '
            'width=(int)%d, height=(int)%d, '
            'format=(string)NV12, framerate=(fraction)%d/1 ! '
            'nvvidconv flip-method=%d ! '
            'video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! '
            'videoconvert ! '
            'video/x-raw, format=(string)BGR ! appsink' % (
                capture_width, capture_height, framerate, flip_method, display_width, display_height))


class GStreamerVideoStream(VideoStream):
    def __init__(self, name="GStreamerVideoStream"):
        # initialize the video camera stream and read the first frame
        # from the stream
        self.stream = cv2.VideoCapture(gstreamer_pipeline(flip_method=0), cv2.CAP_GSTREAMER)
        (self.grabbed, self.frame) = self.stream.read()

        # initialize the thread name
        self.name = name

        # initialize the variable used to indicate if the thread should
        # be stopped
        self.stopped = True

    def start(self):
        # start the thread to read frames from the video stream
        self.stopped = False
        t = Thread(target=self.update, name=self.name, args=())
        t.daemon = True
        t.start()
        return self

    def update(self):
        # keep looping infinitely until the thread is stopped
        while True:
            # if the thread indicator variable is set, stop the thread
            if self.stopped:
                return

            # otherwise, read the next frame from the stream
            (self.grabbed, self.frame) = self.stream.read()

    def read(self):
        # return the frame most recently read
        return self.frame

    def stop(self):
        # indicate that the thread should be stopped
        self.stopped = True

    def __del__(self):
        self.stopped = True
        if hasattr(self, 'stream') and self.stream.isOpened():
            self.stream.release()


def signal_handler(signal, frame):
    if vs and hasattr(vs, 'stream') and vs.stream.isOpened():
        vs.stop()
        vs.stream.release()
    sys.exit(0)


signal.signal(signal.SIGINT, signal_handler)
# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-s", "--server-ip", required=True,
                help="ip address of the server to which the client will connect")
args = vars(ap.parse_args())

# initialize the ImageSender object with the socket address of the
# server
sender = imagezmq.ImageSender(connect_to="tcp://{}:5555".format(
    args["server_ip"]))

# get the host name, initialize the video stream, and allow the
# camera sensor to warmup
rpiName = socket.gethostname()
# vs = VideoStream(usePiCamera=True).start()
vs = GStreamerVideoStream()
if not (vs.stream and vs.stream.isOpened):
    sys.exit(1)

vs.start()
time.sleep(2.0)

while True:
    frame = vs.read()
    sender.send_image(rpiName, frame)

vs.stop()
