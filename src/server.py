import os
import argparse
from PIL import Image
from flask import Flask, request, send_file, url_for, redirect, send_from_directory, jsonify
from werkzeug.utils import secure_filename
import json
from pathlib import Path

from src.infer import SuperGAN

parser = argparse.ArgumentParser()

app = Flask(__name__, static_folder='../webapp/build', static_url_path='/')

UPLOAD_FOLDER = os.environ.get("UPLOAD_FOLDER", os.getcwd() + '/images')

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

MODELS_PATH = os.environ.get("MODELS_PATH")
if not MODELS_PATH:
    raise ValueError("No Models path provided")

device = os.environ.get("DEVICE", 'cpu')

def get_path_for_model(model):
    return MODELS_PATH+'/superGAN_best.pth'


MAX_WIDTH = 1239
MAX_HEIGHT = 1754
def process(image_path, model=None):
    img = Image.open(image_path)
    if img.width > MAX_WIDTH or img.height > MAX_HEIGHT:
        img = img.resize((MAX_WIDTH, MAX_HEIGHT))
    model_path = get_path_for_model(model)
    model = SuperGAN(model_path=model_path, device=device)
    print("Processing ", image_path)
    out = model.process_image(img)
    print("Processed ", image_path)
    out.save(image_path.replace('.png', '_processed.png'))
    model.free()


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


@app.route('/server/upload', methods = ['POST'])
def upload_file():
    f = request.files['file']
    filename = secure_filename(f.filename)
    f.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
    process(os.path.join(app.config['UPLOAD_FOLDER'], filename))

    return redirect(url_for('uploaded_file', filename=filename, _scheme='https', _external=True))


@app.route('/server/models', methods = ['GET'])
def models_list():
    models = Path(MODELS_PATH).glob('*.json')
    result = dict()
    for model in models:
        with open(str(model), 'r') as f:
            data = json.load(f)
            result.update({model.stem: {'name': data['name'],
                                        'description': data['description']
                                        }
                           })

    return jsonify(isError=False, message="Success", statusCode=200, data=result)


@app.route('/')
def index():
    return app.send_static_file('index.html')


