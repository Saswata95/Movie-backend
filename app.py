import os, shutil

from flask import Flask, request, redirect, url_for, send_from_directory, logging, json, jsonify
import werkzeug
from flask_sqlalchemy import SQLAlchemy
from werkzeug.utils import secure_filename
from werkzeug.contrib.fixers import ProxyFix
from flask_celery import make_celery

from flask_restplus import Api, Resource, reqparse

from distutils.dir_util import copy_tree
import sqlite3

import tensorflow as tf
import cv2
import glob
import csv
import pickle
import numpy as np


# Initialize the Flask application
app = Flask(__name__)
#api = Api(app)
app.wsgi_app = ProxyFix(app.wsgi_app)
api = Api(app, version='1.0', title='Movie-Genre API',
    description='A simple Movie Genre Prediction API',
)

# This is the path to the upload directory
app.config['UPLOAD_FOLDER'] = 'uploads/single/'
app.config['UPLOAD_FOLDERS'] = 'uploads/multiple/'
# These are the extension that we are accepting to be uploaded
app.config['ALLOWED_EXTENSIONS'] = set(['jpg', 'jpeg'])

app.config.update(
    CELERY_BROKER_URL='amqp://guest:guest@localhost:5672//',
    CELERY_RESULT_BACKEND='db+sqlite:///results.sqlite'
)

celery = make_celery(app)

# For a given file, return whether it's an allowed type or not
def allowed_file(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1] in app.config['ALLOWED_EXTENSIONS']

#parser = reqparse.RequestParser()
api_parser = reqparse.RequestParser()
api_parser.add_argument('file',type=werkzeug.datastructures.FileStorage, location='form')
#parser.add_argument('formf', type=str, location='form')

@api.route('/upload')
#@api.expect(api_parser)
@api.response(404, 'No File Found')
class ImageUpload(Resource):
    decorators = []

    def post(self):
        #app.logger.info(request.data)
        data = request.files
        if data['file'] == "":
            return '', 404
        photo = data['file']
        app.logger.info(photo)

        if photo and allowed_file(photo.filename):
            filename = secure_filename(photo.filename)
            photo.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            return {
                'data':'',
                'message':'photo uploaded',
                'status':'success'
            }
        return {
            'data':'',
            'message':'Something went wrong',
            'status':'Error'
        }, 404

@api.route('/uploads')
#@api.expect(api_parser)
@api.response(404, 'No Files Found')
class ImagesUpload(Resource):
    decorators=[]

    def post(self):
        uploaded_files = request.files.getlist("file")
        filenames = []
        app.logger.info(len(uploaded_files))
        i=0

        for photos in uploaded_files:
            app.logger.info(photos)
            if photos and allowed_file(photos.filename):
                filename = secure_filename(photos.filename)
                photos.save(os.path.join(app.config['UPLOAD_FOLDERS'], filename))
                filenames.append(filename)
                i+=1
                if i==len(uploaded_files):
                    return {
                    'data':'',
                    'message':'photo uploaded',
                    'status':'success'
                }
        return {
            'data':'',
            'message':'Something went wrong',
            'status':'Error'
        }, 404

@api.route('/predicts')
#@api.expect(api_parser)
@api.response(404, 'No Files Found')
class Predict_Multiple(Resource):
    decorators=[]
    def get(self):
        result = predict_multiple.delay()
        
        return {
            'data':'',
            'message':'success'
        }, 200


@api.route('/fetch')
#@api.expect(api_parser)
@api.response(404, 'No Files Found')
class Fetch_Multiple(Resource):
    decorators=[]
    def get(self):
        pickle_result = open("./static/prediction.pickle", "rb")
        final_result = pickle.load(pickle_result)
        print(final_result)        
        return json.dumps(final_result)

pickle_dataset = open("./static/gen.pickle", "rb")
class_gen = pickle.load(pickle_dataset)

Name = "./static/Movie_Poster_Pred4.model"

def test_img2(filepath):
    global filepaths
    filepaths = glob.glob(filepath)
    IMG_SIZE = 150
    img = []
    for filepat in filepaths:
        img_array = cv2.imread(filepat)
        new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
        new_array = new_array.astype(np.float32)
        img.append(new_array)
    return np.asarray(img)/255.0
@celery.task()
def predict_multiple():
    model_p = tf.keras.models.load_model(Name)
    prediction = model_p.predict_proba([test_img2('./uploads/multiple/*.jpg')])
    pred = np.round(prediction, 2)
    pred = pred.tolist()
    lst_gen = []
    strng_gen = ''
    for i in range(len(pred)):
        diction = dict(zip(class_gen, pred[i]))
        sorted_dict = sorted(diction.values(), reverse=True)
        sorted_dict = sorted_dict[:len(sorted_dict)-5]
        for j in range(len(sorted_dict)):
            sorted_dict[j] *= 100
            for genr, scr in diction.items():
                scr *= 100
                if int(sorted_dict[j]) == 0:
                    continue
                if sorted_dict[j] == sorted_dict[j-1]:
                    continue
                if scr == sorted_dict[j]:
                    strng_gen = strng_gen+str(genr)
                    if j == 2:
                        break
                    else:
                        strng_gen = strng_gen+ ','
                    
        #getting the movie id
        strng_mov=filepaths[i]
        startl = strng_mov.rfind("/")+1
        endl = len(strng_mov)-4
        strng_mov = strng_mov[startl:endl]
        print(strng_mov + ":"+strng_gen)
        predt = {strng_mov: tuple(strng_gen)}
        lst_gen.append(predt)
        strng_gen = ''

    #saving The Result
    pickle_result = open("./static/prediction.pickle", "wb")
    pickle.dump(lst_gen, pickle_result)
    pickle_result.close()
    
    return lst_gen

if __name__ == '__main__':
    app.secret_key='secret123'
    app.run(debug=True)


