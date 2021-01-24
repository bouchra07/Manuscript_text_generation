from __future__ import division
from __future__ import print_function
from flask import Flask, render_template, request, jsonify, redirect
import argparse

import cv2
import editdistance
import os

from DataLoader import DataLoader, Batch
from Model import Model, DecoderType
from SamplePreprocessor import preprocess
import secrets

app = Flask(__name__)
app.secret_key = "super secret key"

fnCharList = '../model/charList.txt'
fnAccuracy = '../model/accuracy.txt'
fnInfer = './static/uploads/test.png'

app.config["UPLOAD_DIRECTORY"] = "static\\uploads"

def infer(model, fnImg):
    img = preprocess(cv2.imread(fnImg, cv2.IMREAD_GRAYSCALE), Model.imgSize)
    batch = Batch(None, [img])
    (recognized, probability) = model.inferBatch(batch, True)
    print('Recognized:', '"' + recognized[0] + '"')
    print('Probability:', probability[0])
    return recognized[0]

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/search', methods=['POST', 'GET'])
def search():

    if request.method == "POST":
        RESULTS_ARRAY = []
        if request.files["image"]:
            parser = argparse.ArgumentParser()
            parser.add_argument('--dump', help='dump output of NN to CSV file(s)', action='store_true')

            args = parser.parse_args()
            decoderType = DecoderType.BestPath
            image = request.files["image"]
            image_dir_name = secrets.token_hex(16)
            image.save(os.path.join(app.config["UPLOAD_DIRECTORY"], image.filename))
            print(image_dir_name)
            print(open(fnAccuracy).read())
            model = Model(open(fnCharList).read(), decoderType, mustRestore=True, dump=args.dump)
            recognized = infer(model, fnInfer)

            return render_template("index.html",image=image.filename, recognized=recognized)
        else:
            return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
