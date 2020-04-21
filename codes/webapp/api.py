import os
# from PIL import Image
from flask import Flask, request, send_file, url_for, redirect, send_from_directory
from werkzeug.utils import secure_filename

app = Flask(__name__, static_folder='build', static_url_path='/')
app.config['UPLOAD_FOLDER'] = 'images'

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

# def rotate(image_path, degrees_to_rotate, saved_location):
#     image_obj = Image.open(image_path)
#     rotated_image = image_obj.rotate(degrees_to_rotate)
#     rotated_image.save(saved_location)
#     rotated_image.show()

@app.route('/api/upload', methods = ['POST'])
def upload_file():
    f = request.files['file']
    filename = secure_filename(f.filename)
    f.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
    # rotate(os.path.join(app.config['UPLOAD_FOLDER'], filename), 90, os.path.join(app.config['UPLOAD_FOLDER'], "rot" + filename))
    return redirect(url_for('uploaded_file', filename=filename))

@app.route('/')
def index():
    return app.send_static_file('index.html')