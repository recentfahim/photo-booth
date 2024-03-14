import os


def filter_image_output(folder_name):
    folder_name = folder_name.split('/')[:-1]
    folder_name = '/'.join(folder_name)
    output_folder_name = folder_name + '/filter_image/'
    create_directory(output_folder_name)
    return output_folder_name


def create_directory(path):
    if not os.path.exists(path):
        os.makedirs(path)