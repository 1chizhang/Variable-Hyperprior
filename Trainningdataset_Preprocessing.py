
import os.path
from pathlib import Path

from PIL import Image
import cv2
import numpy as np
import os
from typing import List

# from torchvision.datasets.folder
IMG_EXTENSIONS = (
    ".jpg",
    ".jpeg",
    ".png",
    ".ppm",
    ".bmp",
    ".pgm",
    ".tif",
    ".tiff",
    ".webp",
)

def collect_images(rootpath: str) -> List[str]:
    return [
        os.path.join(rootpath, f)
        for f in os.listdir(rootpath)
        if os.path.splitext(f)[-1].lower() in IMG_EXTENSIONS
    ]

def preprocessing(imgdir, savedir):
    """

    :param imgdir: input ILSVRC largest 8000 images
    :param savedir: the proprecessed image save dir
    :return:
    Add noise
    "Checkerboard Context Model for Efficient Learned Image Compression"
    Following previous works [5, 6], we add random uniform noise to each of them and then downsample all the images.
    Reshaping
    "VARIATIONAL IMAGE COMPRESSION WITH A SCALE HYPERPRIOR"
    The models were trained on a body of color JPEG images with heights/widths between 3000 and
    5000 pixels, comprising approximately 1 million images scraped from the world wide web. Images
    with excessive saturation were screened out to reduce the number of non-photographic images.
    To reduce existing compression artifacts, the images were further downsampled by a randomized
    factor, such that the minimum of their height and width equaled between 640 and 1200 pixels. Then,
    randomly placed 256x256 pixel crops of these downsampled images were extracted.
    """
    imgdir = Path(imgdir)
    savedir = Path(savedir)
    if not os.path.exists(savedir):
        os.makedirs(savedir)

    if not imgdir.is_dir():
        raise RuntimeError(f'Invalid directory "{imgdir}"')

    img_paths = collect_images(imgdir)
    for imgpath in img_paths:
        img = cv2.imread(imgpath)
        img = np.array((img)).astype(('float64'))
        height, width, channel = img.shape
        ### adding unifor noise
        noise = np.random.uniform(0, 1, (height, width, channel)).astype('float32')
        img += noise
        img = img.astype('uint8')
        if min(width, height)>512:
            img = cv2.resize(img, dsize=((int(width//2), int(height//2))), interpolation=cv2.INTER_CUBIC)
        name = os.path.splitext(os.path.basename(imgpath))[0]
        cv2.imwrite(os.path.join(savedir, name + '.png'), img)

def select_n_images(imgdir, savedir, n):
    """

    :param imgdir: input image dir
    :param savedir: seleted image savingdir
    :param n: the largest n images in the imgdir
    :return:
    """
    import bisect
    import shutil
    import imagesize
    imgdir = Path(imgdir)
    savedir = Path(savedir)
    if not os.path.exists(savedir):
        os.makedirs(savedir)

    if not imgdir.is_dir():
        raise RuntimeError(f'Invalid directory "{imgdir}"')

    img_paths = collect_images(imgdir)
    sizepath = []
    namepath = []
    for imgpath in img_paths:
        width, height = imagesize.get(imgpath)
        size = width*height
        loc = bisect.bisect_left(sizepath, size)
        sizepath.insert(loc, size)
        namepath.insert(loc, imgpath)
        if len(sizepath)>n:
            sizepath.pop(0)
            namepath.pop(0)
    for path in namepath:
        imgname = os.path.basename(path)
        shutil.copyfile(path, os.path.join(savedir, imgname))


if __name__ == '__main__':
    inputimagedir = './dataset/CLIC2021_train' ## original image dataset
    tmpdir = './dataset/tmp' ## temporary image folder
    savedir = './CLIC2021_train_dowmsample' ## preprocessed image folder
    select_n_images(inputimagedir, tmpdir, 8000) ## select 8000 images from ImageNet training dataset, and all image from CLIC training
    preprocessing(tmpdir, savedir)
