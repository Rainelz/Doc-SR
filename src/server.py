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

def get_config_for_model(model):
    weights_path = MODELS_PATH+f'/{model}.pth'
    config_path = MODELS_PATH+f'/{model}.json'
    with open(config_path, 'r') as f:
        config = json.load(f)
    config.pop('name', None)
    config.pop('description', None)
    return weights_path, config


MAX_WIDTH = 1239
MAX_HEIGHT = 1754
def process(image_path, model):
    img = Image.open(image_path)
    #if img.width > MAX_WIDTH or img.height > MAX_HEIGHT:
        #img = img.resize((MAX_WIDTH, MAX_HEIGHT))
    model_path, config = get_config_for_model(model)
    print(f"Processing {image_path} with {model}")
    model = SuperGAN(model_path=model_path, config=config, device=device)
    out = model.process_image(img)
    print("Processed ", image_path)
    out.save(image_path.replace('.png', '_processed.png'))
    model.free()


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


@app.route('/server/upload', methods=['POST'])
def upload_file():
    f = request.files['file']
    model_name = request.form['model']
    if model_name == 'null':
        return "Model name not specified", 404
    filename = secure_filename(f.filename)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    f.save(file_path)
    process(file_path, model_name)

    return redirect(url_for('uploaded_file', filename=filename)) #, _scheme='https', _external=True))


@app.route('/server/models', methods=['GET'])
def models_list():
    models = Path(MODELS_PATH).glob('*.json')
    result = dict()
    for model in models:
        print(model)
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


