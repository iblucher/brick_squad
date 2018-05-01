from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import io
import os
import sys
import json
import urllib3
import multiprocessing
import numpy

from PIL import Image
from tqdm import tqdm
from urllib3.util import Retry

# numpy.random.seed(1)

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

def download_image(fnames_and_urls):
    """
    download image and save its with 90% quality as JPG format
    skip image downloading if image already exists at given path
    :param fnames_and_urls: tuple containing absolute path and url of image
    """
    fname, url = fnames_and_urls
    if not os.path.exists(fname):
        http = urllib3.PoolManager(retries=Retry(connect=3, read=2, redirect=3))
        response = http.request("GET", url)
        image = Image.open(io.BytesIO(response.data))
        image_rgb = image.convert("RGB")
        image_rgb.save(fname, format='JPEG', quality=90)

def parse_dataset(_dataset, _outdir, _num):
    """
    parse the dataset to create a list of tuple containing absolute path and url of image
    :param _dataset: dataset to parse
    :param _outdir: output directory where data will be saved
    :param _max: maximum images to download (change to download all dataset)
    :return: list of tuple containing absolute path and url of image
    """
    _fnames_urls = []
    with open(dataset, 'r') as f:
        data = json.load(f)
        if (num):
            ids = numpy.random.choice(len(data["images"]), num, replace=False)
            print("Downloading {} random images out of {}:".format(num,len(data["images"])))
            for i in range(len(ids)):
                url = data["images"][ids[i]]["url"]
                fname = os.path.join(outdir, "{}.jpg".format(data["images"][ids[i]]["imageId"]))
                _fnames_urls.append((fname, url))
        else:
            for image in data["images"]:
                url = image["url"]
                fname = os.path.join(outdir, "{}.jpg".format(image["imageId"]))
                _fnames_urls.append((fname, url))
    return _fnames_urls


if __name__ == '__main__':
    if len(sys.argv) == 4:
        dataset, outdir, num = sys.argv[1], sys.argv[2], int(sys.argv[3])
    elif len(sys.argv) == 3:
        dataset, outdir, num = sys.argv[1], sys.argv[2], 0
    else:
        print("Usage: python downloader.py file.json path/to/download (max_entries)")
        sys.exit(0)

    if not os.path.exists(outdir):
        os.makedirs(outdir)

    # parse json dataset file
    fnames_urls = parse_dataset(dataset, outdir, num)

    # download data
    pool = multiprocessing.Pool(processes=12)
    with tqdm(total=len(fnames_urls)) as progress_bar:
        for _ in pool.imap_unordered(download_image, fnames_urls):
            progress_bar.update(1)

    sys.exit(1)
