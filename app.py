from flask import Flask, request, render_template, send_from_directory, redirect, url_for
import os
from utils import filter_request, get_image_response_path


app = Flask(__name__)

image_path = 'static'
filter_image_path = os.path.join(image_path, 'filter_image')
original_image_path = os.path.join(image_path, 'original_image')


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
            'whitening_1': 'This filter result of applied of method 1',
            'whitening_2': 'This filter result of applied of method 2',
            'whitening_3': 'This filter result of applied of method 3',
            'whitening_4': 'This filter result of applied of method 4',
            'cartoon': 'Creating a filter that transforms images into cartoon-like representations, with exaggerated features and simplified details.',
            'pencil_sketch': 'Implementing a filter that converts images into pencil sketches, mimicking the appearance of hand-drawn sketches.',
            'background_remove': 'Implementing a feature that allows users to remove the background from images, leaving only the foreground subject.'
        }
    }


@app.route('/apply-filter/', methods=['POST'])
def apply_filter():
    filter_name = request.form.get('filter_name')
    image = request.files['image']

    response = filter_request(filter_name, image)
    if response.get('success'):
        image_name = response.get('image_name')
        context = get_image_response_path(image_name)

        return context

    return response


@app.route('/filter/', methods=['POST'])
def filter_apply():
    filter_name = request.form.get('filter_name')
    image = request.files['image']

    response = filter_request(filter_name, image)

    if response.get('success'):
        return redirect(url_for('result', image_name=response.get('image_name')))

    return response


@app.route('/')
def home():
    return render_template('home.html')


@app.route('/result')
def result():
    image_name = request.args.get('image_name')
    image_path = get_image_response_path(image_name)

    return render_template('result.html', **image_path)


if __name__ == '__main__':
    app.run()
