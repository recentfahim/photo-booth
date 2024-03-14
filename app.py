from flask import Flask, request, render_template, send_from_directory, redirect, url_for
from PIL import Image
from filter.pencil_sketch import pencil_sketch
from filter.cantoon_filter import cantoon_filter
from filter.remove_bg import remove_background
import uuid
import os


app = Flask(__name__)

UPLOAD_FOLDER = 'images'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/test-api', methods=['GET'])
def test_api():
    return {
        'message': 'The Photo Booth API is working perfectly',
        'success': True
    }


@app.route('/available-filter/', methods=["GET"])
def available_filters():
    return {
        'message': 'These are available filter you can apply. Please enter the filter name in the given api endpoint',
        'success': True,
        'data': {
            'cartoon': 'Creating a filter that transforms images into cartoon-like representations, with exaggerated features and simplified details.',
            'pencil_sketch': 'Implementing a filter that converts images into pencil sketches, mimicking the appearance of hand-drawn sketches.',
            'background_remove': 'Implementing a feature that allows users to remove the background from images, leaving only the foreground subject.'
        }
    }


@app.route('/apply-filter/', methods=['POST'])
def apply_filter():
    filter_name = request.form.get('filter_name')
    image = request.files['image']

    if not filter_name and not image:
        if not image:
            return {
                'message': 'Image is not found',
                'success': False
            }
        if not filter_name:
            return {
                'message': 'You did not enter any filter',
                'success': False
            }
    else:
        image_name = str(uuid.uuid4()) + '.' + image.filename.split('.')[-1]
        if not os.path.exists(app.config['UPLOAD_FOLDER']):
            os.makedirs(app.config['UPLOAD_FOLDER'])

        image.save(os.path.join(app.config['UPLOAD_FOLDER'], image_name))

        if filter_name == 'cartoon':
            cantoon_filter(app.config['UPLOAD_FOLDER'], image_name)
        elif filter_name == 'pencil_sketch':
            pencil_sketch(app.config['UPLOAD_FOLDER'], image_name)
        elif filter_name == 'background_remove':
            remove_background(app.config['UPLOAD_FOLDER'], image_name)
        else:
            return {
                'message': 'You entered wrong filters. Please visit /available-filter/ to see the correct name',
                'success': False
            }

    return {
        'message': filter_name,
        'success': True
    }


@app.route('/')
def home():
    return render_template('home.html')


if __name__ == '__main__':
    app.run()
