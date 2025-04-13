'''
This script downloads the ABI-1 Corpus dataset and the WavLM model,
which are used for voice similarity analysis.
Download the ABI-1 Corpus dataset from Google Drive and the WavLM model from Hugging Face.
This script takes several minutes to run
'''


import os
import shutil
import zipfile
import gdown
from transformers import AutoModel, Wav2Vec2FeatureExtractor
from utils import timer


@timer
def create_folders():
    '''Create the data and model folders if they don't exist.'''

    data_folder = os.path.join(
        os.path.dirname(__file__),
        'data')
    if not os.path.exists(data_folder):
        os.makedirs(data_folder, exist_ok=True)

    model_folder = os.path.join(
        os.path.dirname(__file__),
        'model')
    if not os.path.exists(model_folder):
        os.makedirs(model_folder, exist_ok=True)

@timer
def download_data():
    '''Download the ABI-1 Corpus dataset from Google Drive and extract it.'''

    # google drive id
    file_id = '18FWBn4B6gQifOtf1C9JCQv4Lrs8C1uvu'

    # download the file
    data_download_folder = os.path.join(
        os.path.dirname(__file__),
        'data',
        'download',
        'ABI-1_Corpus')

    if not os.path.exists(data_download_folder):
        os.makedirs(data_download_folder, exist_ok=True)

    download_path = os.path.join(data_download_folder, 'ABI-1_Corpus.zip')
    gdown.download(f'https://drive.google.com/uc?id={file_id}', download_path, quiet=False)

    # open the zip
    data_raw_folder = os.path.join(
        os.path.dirname(__file__),
        'data',
        'raw',
        'ABI-1_Corpus')
    if not os.path.exists(data_raw_folder):
        os.makedirs(data_raw_folder, exist_ok=True)

    with zipfile.ZipFile(download_path, 'r') as zip_ref:
        zip_ref.extractall(data_raw_folder)

@timer
def download_model():
    '''Download the WavLM model from Hugging Face.'''

    # model folder at microsoft and corresponding local folder
    model_name = 'microsoft/wavlm-base-plus-sv'
    model_folder = os.path.join(
        os.path.dirname(__file__),
        'model')

    # empty the model folder
    if os.path.exists(model_folder):
        print(f'Removing existing model folder: {model_folder}')
        shutil.rmtree(model_folder)

    # download the model
    Wav2Vec2FeatureExtractor.from_pretrained(model_name, cache_dir=model_folder)
    AutoModel.from_pretrained(model_name, cache_dir=model_folder)



if __name__ == '__main__':
    print('Starting folder creation...')
    create_folders()
    print('Folder creation complete.')

    print('Starting data download...')
    download_data()
    print('Data download complete.')

    print('Starting model download...')
    download_model()
    print('Model download complete.')
