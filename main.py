from genericpath import exists
from monocularDepth import MonocularDepth
from image3D import Image3D
from gif3D import Gif3D
import numpy as np
from concurrent.futures import ThreadPoolExecutor, wait
import socketio
import os
import cv2
import shutil

sio = socketio.Client()

class Processor3D(socketio.ClientNamespace):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.md = MonocularDepth()
        self.i3d = Image3D('argument.yml')
        self.gif = Gif3D()
        self.pool = ThreadPoolExecutor(max_workers=1)

    def on_connect(self):
        pass

    def on_disconnect(self):
        pass

    def on_check_person(self, img_id, img_binary) -> np.array:
        nparr = np.frombuffer(img_binary, np.uint8)
        img_np = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        print('Segmenting...')
        _, img_masked = self.gif.segmentation(img_np)
        self.emit('result_check_person', data=(img_id, img_masked.any()))
        # if not img_masked.any():
        #     print('Return False')
        #     return False
        # result = cv2.imencode('.jpg', img_masked, [int(cv2.IMWRITE_JPEG_QUALITY), 100])[1].tostring()
        # print('Return image')
        # return result

    def __run(self, src):
        self.md.run_monocularDepth(src + '/image', src + '/depth')
        self.i3d.run_3dimage(src)
        self.gif.run_3dgif(src)

    def on_run(self, img_id, img_binary):
        src = f'/tmp/{img_id}'
        os.makedirs(src + '/image', exist_ok=True)
        os.makedirs(src + '/depth', exist_ok=True)

        with open(src + '/image/' + str(img_id) + '.jpg', 'wb') as fimg:
            fimg.write(img_binary)

        ex : Exception = self.pool.submit(self.__run, src).exception()
        
        if ex is None:
            final_3dvideo = src + '/' + str(img_id) + '.mp4'
            if exists(final_3dvideo):
                with open(final_3dvideo, 'rb') as res:
                    self.emit('result_3dvideo', data=(img_id, res.read()))
        else:
            self.emit('result_3dvideo', data=(img_id, False))
            raise ex
        # shutil.rmtree(src)

if __name__ == '__main__':
    sio.register_namespace(Processor3D('/processor3D'))
    sio.connect('http://213.37.41.31:80', namespaces='/processor3D')
    sio.wait()