import requests
import time
import os
import numpy as np

subscription_key = "d7b326d2f5fa4ff4b3f329acedc59641"
assert subscription_key

vision_base_url = "https://westcentralus.api.cognitive.microsoft.com/vision/v2.0/"

text_recognition_url = vision_base_url + "recognizeText"

headers = {'Ocp-Apim-Subscription-Key': subscription_key, 'Content-Type': 'application/octet-stream'}

params  = {'mode': 'Printed'}

def abs_join(*args):
	return os.path.join(os.path.split(os.path.realpath(__file__))[0], *args)

def parse_one_detection(polygon):
    points = np.array(polygon[0])
    word = polygon[1]
    p1 = points[0:2]
    p2 = points[2:4]
    p3 = points[4:6]
    w = int(np.linalg.norm(p1 - p2))
    h = int(np.linalg.norm(p2 - p3))
    size = w * h

    return '[' + str(h) + '|' + str(size) + ']' + word


def m_post(*args, **kwargs):
    while True:
        try:
            return requests.post(*args, **kwargs)
        except:
            time.sleep(5)

def m_get(*args, **kwargs):
    while True:
        try:
            return requests.get(*args, **kwargs)
        except:
            time.sleep(5)

def do_a_image(image_path):
    data = open(image_path, 'rb').read()

    response = m_post(text_recognition_url, headers=headers, params=params, data=data)
    response.raise_for_status()

    operation_url = response.headers["Operation-Location"]

    analysis = {}
    poll = True
    while (poll):
        response_final = m_get(
            response.headers["Operation-Location"], headers=headers)
        analysis = response_final.json()
        time.sleep(1)
        if ("recognitionResult" in analysis):
            poll= False 
        if ("status" in analysis and analysis['status'] == 'Failed'):
            poll= False

    polygons=[]
    if ("recognitionResult" in analysis):
        # Extract the recognized text, with bounding boxes.
        polygons = [(line["boundingBox"], line["text"])
            for line in analysis["recognitionResult"]["lines"]]

    result = ''
    for polygon in polygons:
        result += parse_one_detection(polygon) + ''

    return result[:-1]

datadir = abs_join('.', 'data')
# origin_dir = abs_join('..', 'download', 'opencv-text-recognition', 'images')
origin_dir = abs_join(datadir, 'ori_ocr')
cartoon_dir = abs_join(datadir, 'car_ocr')
recover_dir = abs_join(datadir, 'rec_ocr')

origin_img_names = sorted(os.listdir(origin_dir))

with open(abs_join(datadir, 'ms_result.csv'), 'w') as f:
    for origin_img_name in origin_img_names:
        cartoon_img_name = origin_img_name.split('.')[0][-7] + '_fake_A.png'
        recover_img_name = origin_img_name.split('.')[0][-7] + '_rec_B.png'

        result = origin_img_name + ', '
        ori_results = do_a_image(os.path.join(origin_dir, origin_img_name))
        car_results = do_a_image(os.path.join(cartoon_dir, cartoon_img_name))
        rec_results = do_a_image(os.path.join(recover_dir, recover_img_name))

        result += ori_results + ', ' + car_results + ', ' + rec_results + '\n'

        f.writelines([result])
        print(result)
