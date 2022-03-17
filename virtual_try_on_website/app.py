import os
import json
from flask import Flask, request, render_template, flash, redirect, url_for
from werkzeug.utils import secure_filename
import urllib.request

app = Flask(__name__)

UPLOAD_FOLDER = 'static/uploads/'
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])

app.secret_key = "secret key"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
image_save_paths = {}

# Helper funtions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def save_customer_images(image_profile):
    if image_profile not in request.files:
        flash('No file part')
        return redirect(request.url)

    image = request.files[image_profile]

    if image.filename == '':
        flash('No image selected for uploading')
        return redirect(request.url)

    if image and allowed_file(image.filename):
        filename = secure_filename(image.filename)
        save_path = os.path.join(app.config['UPLOAD_FOLDER'] + f'images/{image_profile}/', filename)
        image.save(save_path)
        image_save_paths[image_profile] = save_path
        print('upload_image filename: ' + filename)
    else:
        flash('Allowed image types are - png, jpg, jpeg, gif')
        return redirect(request.url)

def save_customer_information():
    customer_info = {
        'name': request.form['name'], 
        'height': request.form['height'],
        'front': image_save_paths['front'],
        'side': image_save_paths['side']
    }
    str_name = customer_info['name'].replace(' ', '_')
    save_path = os.path.join(app.config['UPLOAD_FOLDER'] + 'customer_info/', f'{str_name}.json')
    with open(save_path, "w") as outfile:
        json.dump(customer_info, outfile)

def check_paths():
    for image_profile in ['front', 'side']:
        profile_path = os.path.join(app.config['UPLOAD_FOLDER'], f'images/{image_profile}/')
        if not os.path.exists(profile_path):
            print('Creating', profile_path)
            os.makedirs(profile_path)

    info_path = os.path.join(app.config['UPLOAD_FOLDER'], 'customer_info/')
    if not os.path.exists(info_path):
        print('Creating', info_path)
        os.makedirs(info_path)

# Website functions
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/', methods=['POST'])
def upload_image():
    print(1)
    check_paths()
    for profile in ['front', 'side']:
        save_customer_images(profile)
    flash('Images successfully saved')

    print(image_save_paths)
    save_customer_information()
    flash('Customer information successfully saved')
    return render_template('index.html')
    

# @app.route('/display/<filename>')
# def display_image(filename):
#     #print('display_image filename: ' + filename)
#     return redirect(url_for('static', filename='uploads/' + filename), code=301)

if __name__ == "__main__":
    app.run()
