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
import page
import words
from PIL import Image
import cv2

app = Flask(__name__)
app.secret_key = "super secret key"

fnCharList = '../model/charList.txt'
fnAccuracy = '../model/accuracy.txt'
fnInfer = './segmented/segment'

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
            print(image.filename)

            image2 = cv2.cvtColor(cv2.imread("./static/uploads/"+image.filename), cv2.COLOR_BGR2RGB)
            crop = page.detection(image2)
            boxes = words.detection(crop)
            lines = words.sort_words(boxes)

            # Saving the bounded words from the page image in sorted way
            i = 0
            

            open(fnAccuracy).read()
            model = Model(open(fnCharList).read(), decoderType, mustRestore=True, dump=args.dump)
            
            for line in lines:
                text = crop.copy()
                for (x1, y1, x2, y2) in line:
                    # roi = text[y1:y2, x1:x2]
                    save = Image.fromarray(text[y1:y2, x1:x2])
                    # print(i)
                    save.save("segmented/segment" + str(i) + ".png")
                    i += 1

            path, dirs, files = next(os.walk("./segmented"))
            file_count = len(files)

            recognized = []
            for i in range(file_count):
                recognized.append(infer(model,"./segmented/segment" + str(i) + ".png"))


            # print(open(fnAccuracy).read())
            # model = Model(open(fnCharList).read(), decoderType, mustRestore=True, dump=args.dump)
            # recognized = infer(model, fnInfer)

            return render_template("index.html",image=image.filename, words=recognized)
        else:
            return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
