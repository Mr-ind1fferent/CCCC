import subprocess
import threading

from PyQt5.QtCore import pyqtSignal, QObject


class TrackCmdExecutor(threading.Thread, QObject):
    startedTrack = pyqtSignal()
    gotFrameCount = pyqtSignal(int)
    gotTempImg = pyqtSignal()
    updateProgress = pyqtSignal(int)
    finishedTrack = pyqtSignal()
    savedVideo = pyqtSignal()

    def __init__(self, order: str, thread_lock: threading.Lock):
        threading.Thread.__init__(self)
        QObject.__init__(self)
        print(order)
        self.isExecuting = False
        self.order = order
        self.threadLock = thread_lock
        self.cmd = subprocess.Popen(self.order, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        self.frameCount = 999999

    def run(self) -> None:
        self.isExecuting = True
        for i in iter(self.cmd.stdout.readline, 'b'):
            if not i:
                break

            try:
                msg = i.decode('utf8')
                print(msg)

                if msg[:10] == 'temp_image':
                    self.gotTempImg.emit()
                elif msg[0] == '[':
                    if 'Processing frame' in msg:
                        self.updateProgress.emit(int(msg.split()[-3]) / self.frameCount * 100)
                    elif 'Length of' in msg:
                        self.frameCount = int(msg.split()[-2])
                        self.gotFrameCount.emit(self.frameCount)
                    elif 'MOT results' in msg:
                        self.finishedTrack.emit()
                    elif 'Save video' in msg:
                        self.savedVideo.emit()
                    elif 'Starting tracking' in msg:
                        self.startedTrack.emit()

            except UnicodeDecodeError:
                print('DECODE ERROR')
            self.isExecuting = False

    def terminate(self):
        if self.is_alive():
            if self.threadLock.locked():
                self.threadLock.release()
            print(self.getName(), "命令行终止...")
            self.cmd.terminate()
