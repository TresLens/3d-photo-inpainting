from monocularDepth import MonocularDepth
from image3D import Image3D
from gif3D import Gif3D

if __name__ == '__main__':
    # md = MonocularDepth().run_monocularDepth("C:/tmp/awdfafd/image", "C:/tmp/awdfafd/depth")

    # i3d = Image3D('argument.yml').run_3dimage("C:/tmp/awdfafd")

    Gif3D().run_3dgif("C:/tmp/awdfafd")