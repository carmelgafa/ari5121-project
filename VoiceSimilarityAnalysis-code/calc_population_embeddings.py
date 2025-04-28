'''
This script will go through all the speakers and calculate
the embeddings for all that files, in order to facilitate the 
comparison of the voices.
'''


import os
import re
import pickle
import torch
import pandas as pd
import numpy as np
from transformers import Wav2Vec2FeatureExtractor, WavLMForXVector
from utils import timer


@timer
def generate_embeddings_information(attempt_gpu:bool=False):
    '''
    Goes through all the speakers and calculates
    the embeddings for all their files, using GPU for acceleration.
    '''

    device = torch.device("cuda" if attempt_gpu and torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    embeddings_dict = {}

    local_model_folder = os.path.join(
        os.path.dirname(__file__),
        'model',
        'models--microsoft--wavlm-base-plus-sv',
        'snapshots',
        'feb593a6c23c1cc3d9510425c29b0a14d2b07b1e')

    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(local_model_folder)
    model = WavLMForXVector.from_pretrained(local_model_folder).to(device)
    model.eval()

    cleansed_folder = os.path.join(os.path.dirname(__file__), 'data', 'preprocessed')
    accents = os.listdir(cleansed_folder)

    for accent in accents:
        accent_path = os.path.join(cleansed_folder, accent)
        genders = os.listdir(accent_path)

        for gender in genders:
            gender_path = os.path.join(accent_path, gender)
            speakers = os.listdir(gender_path)

            for speaker in speakers:
                speaker_path = os.path.join(gender_path, speaker)

                for filename in os.listdir(speaker_path):
                    if re.fullmatch(r'shortpassage.*\.npy', filename):
                        print(accent, gender, speaker, filename)


                        audio_array = np.load(os.path.join(speaker_path, filename))

                        target_sr = 16000
                        # Extract features
                        inputs = feature_extractor(
                            [audio_array],
                            sampling_rate=target_sr,
                            return_tensors='pt',
                            padding=True).to(device)

                        with torch.no_grad():
                            embeddings = model(**inputs).embeddings
                            embeddings = torch.nn.functional.normalize(embeddings, dim=-1)

                        speaker_id = f'{accent}-{gender}-{speaker}'

                        if speaker_id in embeddings_dict:
                            embeddings_dict[speaker_id] = torch.cat(
                                [embeddings_dict[speaker_id], embeddings],
                                dim=0)
                        else:
                            embeddings_dict[speaker_id] = embeddings

    # Move embeddings to CPU for saving
    df = pd.DataFrame([
        {'Speaker': speaker, 'Embedding': emb}
        for speaker, emb in embeddings_dict.items()
    ])

    embedding_folder = os.path.join(os.path.dirname(__file__), 'data', 'embeddings')
    os.makedirs(embedding_folder, exist_ok=True)

    with open(os.path.join(embedding_folder, 'embeddings.pkl'), 'wb') as f:
        pickle.dump(df, f)


def check_embeddings_count():
    '''
    Check the number of embeddings for each speaker.
    '''
    df = pd.read_pickle(os.path.join(
        os.path.dirname(__file__),
        'data',
        'embeddings',
        'embeddings.pkl'))

    print(df.head(1)['Embedding'][0].shape)


if __name__ == '__main__':
    print('Calculating embeddings for all speakers...')
    generate_embeddings_information()

    # check_embeddings_count()
