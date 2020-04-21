import os
import argparse
import sys

sys.path.append("../codes") 

from PIL import Image
from flask import Flask, request, send_file, url_for, redirect, send_from_directory
from werkzeug.utils import secure_filename

from infer import SuperGAN

app = Flask(__name__, static_folder='build', static_url_path='/')
app.config['UPLOAD_FOLDER'] = 'images'

MODELS_PATH = os.environ.get("MODELS_PATH")
if not MODELS_PATH:
    raise ValueError("No Models path provided")

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

def get_path_for_model(model):
    return MODELS_PATH+'/superGAN_best.pth'

def process(image_path, model=None):
    model_path = get_path_for_model(model)
    model = SuperGAN(model_path=model_path)
    print("Processing ", image_path)
    img = Image.open(image_path)
    out = model.process_image(img)
    print("Processed ", image_path)
    out.save(image_path.replace('.png', '_processed.png'))
    del model

@app.route('/api/upload', methods = ['POST'])
def upload_file():
    f = request.files['file']
    filename = secure_filename(f.filename)
    f.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
    process(os.path.join(app.config['UPLOAD_FOLDER'], filename))
    return redirect(url_for('uploaded_file', filename=filename))

@app.route('/')
def index():
    return app.send_static_file('index.html')


