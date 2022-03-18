import os
import json
from flask import Flask, request, render_template, flash, redirect, url_for
from werkzeug.utils import secure_filename
import urllib.request

app = Flask(__name__)

UPLOAD_FOLDER = 'static/uploads/'
RESOURCES_FOLDER = 'static/resources/'
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gifs'])

app.secret_key = "secret key"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESOURCES_FOLDER'] = RESOURCES_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

image_save_paths = {}
test_image = RESOURCES_FOLDER + '1449980.png'

# Helper funtions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def get_extension(filename):
    return filename.rsplit('.', 1)[1].lower()

def save_customer_image(image_profile):
    if image_profile not in request.files:
        flash('No file part')
        return redirect(request.url)

    image = request.files[image_profile]

    if image.filename == '':
        flash('No image selected for uploading')
        return redirect(request.url)

    if image and allowed_file(image.filename):
        file_extension = get_extension(image.filename)
        filename = secure_filename(image.filename)
        save_path = os.path.join(app.config['UPLOAD_FOLDER'] + f'images/{image_profile}.{file_extension}')
        image.save(save_path)
        image_save_paths[image_profile] = save_path
        print('upload_image filename: ' + filename)
    else:
        flash('Allowed image types are - png, jpg, jpeg, gif')
        return redirect(request.url)

def save_customer_information():
    customer_info = {
        'height': request.form['height'],
        'front': image_save_paths['front'],
        'side': image_save_paths['side']
    }
    save_path = os.path.join(app.config['UPLOAD_FOLDER'] + 'customer_info/customer_info.json')
    with open(save_path, "w") as outfile:
        json.dump(customer_info, outfile)

def check_paths():
    images_path = os.path.join(app.config['UPLOAD_FOLDER'], f'images/')
    if not os.path.exists(images_path):
        print('Creating', images_path)
        os.makedirs(images_path)

    info_path = os.path.join(app.config['UPLOAD_FOLDER'], 'customer_info/')
    if not os.path.exists(info_path):
        print('Creating', info_path)
        os.makedirs(info_path)

def calculate_size():
    flash('Hello beautiful, we think this product in a size XXS/XS would look amazing on you!')


# Website functions
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/customer_size', methods=['POST'])
def upload_image():
    check_paths()

    # Save Image Locally
    save_customer_image('front')
    save_customer_image('side')
    # flash('Images successfully saved')

    # Save information
    save_customer_information()
    # flash('Customer information successfully saved')

    calculate_size()
    return render_template('customer_info.html', product_image=test_image)

@app.route('/customer_info', methods=['POST'])
def get_product_info():
    print(test_image)
    return render_template('customer_info.html', product_image=test_image)

@app.route('/display/<filename>')
def display_image(filename):
    #print('display_image filename: ' + filename)
    return redirect(url_for('static', filename=test_image), code=301)

if __name__ == "__main__":
    app.run()

    

# @app.route('/display/<filename>')
# def display_image(filename):
#     #print('display_image filename: ' + filename)
#     return redirect(url_for('static', filename='uploads/' + filename), code=301)

if __name__ == "__main__":
    app.run()
