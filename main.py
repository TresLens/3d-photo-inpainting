from monocularDepth import MonocularDepth
from image3D import Image3D
from gif3D import Gif3D

if __name__ == '__main__':
    md = MonocularDepth()
    i3d = Image3D('argument.yml')
    gif = Gif3D()

    for inp in ['inp1', 'inp2', 'inp3', 'inp4']:
        src = f'/content/{inp}'

        md.run_monocularDepth(src + '/image', src + '/depth')
        i3d.run_3dimage(src)
        gif.run_3dgif(src)