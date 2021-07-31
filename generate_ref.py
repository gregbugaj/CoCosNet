
import os
import argparse
from typing import OrderedDict
import numpy as np
import cv2
from tqdm import tqdm
from scipy.spatial import distance

import multiprocessing as mp
from concurrent.futures.thread import ThreadPoolExecutor

def process(pool, images, src_path, out_rev):
    with ThreadPoolExecutor(max_workers=mp.cpu_count()//2) as executor:      
        executor.submit(process, pool, images, src_path, rev) 


def main(path, output):
    ext = {'.JPG', '.JPEG', '.PNG', '.TIF', 'TIFF'}

    images = []
    pool = {}
    for root, dirs, files in os.walk(path):
        print('loading ' + root)
        for file in files:
            file_path = os.path.join(root, file)
            if os.path.splitext(file)[1].upper() in ext:
                images.append(file_path)
                img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
                # img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
                pool[file_path]  = img

    images = sorted(images)

    for i, src_path in enumerate(tqdm(images)):
        src_img = pool[src_path]
        scores = {}
        for dst_path in images:
            dst_img = pool[dst_path]
            v = np.array(src_img)
            u = np.array(dst_img)
            v =  v.flatten()
            u =  u.flatten()

            cosine_similarity = 1 - distance.cosine(u, v)
            scores[dst_path] = cosine_similarity
            # print(f'    > {dst_path} : {cosine_similarity}')

        ordered = {k: v for k, v in sorted(scores.items(), key=lambda item: item[1])}
        rev = list(ordered.items())
        rev.reverse()
        rev = rev[1:4]# skip first items(own id, and get the next K items are our reference values)
        
        # for item in rev[1:4]:
        #     rev.append(item)

        with open(output,'a') as flist:
            try:
                name = src_path.split("/")[-1]
                names =  [(k.split("/")[-1]) for k,v in rev]
                formated = ",".join(names)

                flist.write(name)
                flist.write(',')
                flist.write(formated)
                flist.write('\n')
            except Exception as e:
                raise e
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, help='path to the dataset')
    parser.add_argument('--output', type=str, help='path to the ref file')
    args = parser.parse_args()

    main('/home/greg/dev/pytorch-CycleGAN-and-pix2pix/datasets/HCFA02/ready/testA', '/home/greg/dev/pytorch-CycleGAN-and-pix2pix/datasets/HCFA02/ready/testa_ref.txt')
    # main(args.path, args.output)

# /home/greg/dev/unet-denoiser/data/test/image