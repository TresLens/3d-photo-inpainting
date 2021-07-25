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

BACKEND_URL = 'http://213.37.41.31'
TOKEN = '1vfdGKfSmxMZq6fEnhaQkzoDt6x_2SaJpcYpEjCRYDkgGBGSS'

app = Flask(__name__)

def register_ngrok_address(ngrok_url):
    requests.post(BACKEND_URL + '/register_colab', data={'address': ngrok_url})

run_with_ngrok(app, register_ngrok_address, TOKEN)  # Start ngrok when app is run

md = MonocularDepth()
i3d = Image3D('argument.yml')
gif = Gif3D()
pool = ThreadPoolExecutor(max_workers=1)

@app.route("/check_person", methods=['POST'])
def check_person():
    img_binary = request.files['img_binary'].read()

    nparr = np.frombuffer(img_binary, np.uint8)
    img_np = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    print('Segmenting...')
    _, img_masked = gif.segmentation(img_np)
    return {'result': bool(img_masked.any())}

def run3D(src, img_id):
    url = BACKEND_URL + '/result_3dvideo/'+img_id
    try:
        md.run_monocularDepth(src + '/image', src + '/depth')
        i3d.run_3dimage(src)
        gif.run_3dgif(src)

        final_3dvideo = src + '/' + str(img_id) + '_output.mp4'
        files = {'final_video': open(final_3dvideo, 'rb').read()}
        requests.post(url, files=files)
    except:
        requests.post(url, data={'result': False})
    finally:
        shutil.rmtree(src)


@app.route("/run/<img_id>", methods=['POST'])
def run(img_id):
    img_binary = request.files['img_binary'].read()

    src = f'/tmp/{img_id}'
    os.makedirs(src + '/image', exist_ok=True)
    os.makedirs(src + '/depth', exist_ok=True)

    with open(src + '/image/' + str(img_id) + '.jpg', 'wb') as fimg:
        fimg.write(img_binary)

    pool.submit(run3D, src, img_id)

    return {'result': True}

if __name__ == '__main__':
    app.run()