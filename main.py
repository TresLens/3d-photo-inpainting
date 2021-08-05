from genericpath import exists

from flask.helpers import make_response
from monocularDepth import MonocularDepth
from image3D import Image3D
from gif3D import Gif3D
import numpy as np
from concurrent.futures import ThreadPoolExecutor, wait
import os, cv2, shutil, requests
from flask import Flask, request
from flask_ngrok import run_with_ngrok
from threading import Timer
import IPython
import traceback
  
BACKEND_URL = os.environ['BACKEND_URL']
TOKEN = '1vfdGKfSmxMZq6fEnhaQkzoDt6x_2SaJpcYpEjCRYDkgGBGSS'

app = Flask(__name__)

def shut_down():
    print('SHUTTING DOWN COLAB')
    IPython.display.Javascript("console.log('SHUT DOWN')")

thread_timer = Timer(60*30, shut_down)

def reset_timer():
    global thread_timer

    if not thread_timer.isAlive():
        thread_timer.start()
    else:
        thread_timer.cancel()
        thread_timer = Timer(60*30, shut_down)

def register_ngrok_address(ngrok_url):
    requests.post(BACKEND_URL + '/register_colab', data={'address': ngrok_url})
    reset_timer()

run_with_ngrok(app, register_ngrok_address, TOKEN)  # Start ngrok when app is run

md = MonocularDepth()
i3d = Image3D('argument.yml')
gif = Gif3D()
pool = ThreadPoolExecutor(max_workers=1)

@app.route("/check_person", methods=['POST'])
def check_person():
    reset_timer()
    img_binary = request.files['img_binary'].read()

    nparr = np.frombuffer(img_binary, np.uint8)
    img_np = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    print('Segmenting...')
    _, img_masked = gif.segmentation(img_np)
    return {'result': bool(img_masked.any())}

@app.route("/live", methods=['GET'])
def liveness():
    return {'result': True}

def run3D(src, user_id, img_id):
    url = BACKEND_URL + f'/result_3dvideo/{user_id}/' + img_id
    try:
        md.run_monocularDepth(src + '/image', src + '/depth')
        i3d.run_3dimage(src)
        gif.run_3dgif(src)

        final_3dvideo = src + '/' + str(img_id) + '.mp4'
        files = {'final_video': open(final_3dvideo, 'rb').read()}
        requests.post(url, files=files)
    except Exception as e:
        tr = traceback.format_exc()
        requests.post(url, data={'result': False, 'traceback': tr, 'exception': str(e)})
    finally:
        shutil.rmtree(src)


@app.route("/run/<user_id>/<img_id>", methods=['POST'])
def run(user_id, img_id):
    reset_timer()
    img_binary = request.files['img_binary'].read()

    src = f'/tmp/{img_id}'
    os.makedirs(src + '/image', exist_ok=True)
    os.makedirs(src + '/depth', exist_ok=True)

    with open(src + '/image/' + str(img_id) + '.jpg', 'wb') as fimg:
        fimg.write(img_binary)

    pool.submit(run3D, src, user_id, img_id)

    return {'result': True}

if __name__ == '__main__':
    app.run()