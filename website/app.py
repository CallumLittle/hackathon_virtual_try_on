import os
import json
from flask import Flask, request, render_template, flash, redirect, url_for
from werkzeug.utils import secure_filename
import urllib.request
from size_prediction import get_size_by_photos, save_segmented_images, product_size_recommender

app = Flask(__name__)

UPLOAD_FOLDER = 'static/uploads/'
RESOURCES_FOLDER = 'static/resources/'
SEGMENTED_IMAGES_FOLDER = 'static/image_files/'
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gifs'])

app.secret_key = os.urandom(12) #"secret key"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESOURCES_FOLDER'] = RESOURCES_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

save_images = False
image_save_paths = {}
customer_info = {}
test_image = RESOURCES_FOLDER + '1449980.png'
segmented_images = {
    'front': SEGMENTED_IMAGES_FOLDER + 'front_segmented.jpg',
    'side': SEGMENTED_IMAGES_FOLDER + 'side_segmented.jpg'
}

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
        image_save_paths[image_profile] = save_path
        if save_images:
            image.save(save_path)
            print('Saving ' + filename)
    else:
        flash('Allowed image types are - png, jpg, jpeg, gif')
        return redirect(request.url)
    return save_path

def save_size_information():
    customer_info['height'] = request.form['height'],
    customer_info['front'] = image_save_paths['front'],
    customer_info['side'] = image_save_paths['side']
    save_path = os.path.join(app.config['UPLOAD_FOLDER'] + 'customer_info.json')
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
    front_segmented, side_segmented, waist_in_cm = get_size_by_photos(image_save_paths['front'], image_save_paths['side'], customer_info['height'])
    if save_images:
        save_segmented_images(front_segmented, side_segmented)
    size = product_size_recommender(waist_in_cm)
    print(f'Detected waist size: {waist_in_cm:.2f}cm')
    flash(f'Hello beautiful, we think this product in a size {size} would look amazing on you!')


# Website functions
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/customer_size', methods=['POST'])
def upload_image():
    check_paths()
    save_customer_image('front')
    save_customer_image('side')
    save_size_information()
    calculate_size()
    
    return render_template('customer_info.html', 
        product_image=test_image,
        front_segmented = segmented_images['front'],
        side_segmented = segmented_images['side'])

@app.route('/customer_info', methods=['POST'])
def get_product_info():
    return render_template('customer_info.html', product_image=test_image)

@app.route('/display/<filename>')
def display_image(filename):
    return redirect(url_for('static', filename=test_image), code=301)

if __name__ == "__main__":
    app.run()