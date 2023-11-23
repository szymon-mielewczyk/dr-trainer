import requests
import argparse

parser = argparse.ArgumentParser(description='request prep')
parser.add_argument('img_path', metavar='P', type=str, help='path to image')
args = parser.parse_args()
IMG_PATH = args.img_path

resp = requests.post("http://localhost:5000/predict",
                     files={"file": open(IMG_PATH,'rb')})


print(resp.json())