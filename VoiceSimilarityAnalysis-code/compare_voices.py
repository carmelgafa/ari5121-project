'''
This script contains scripts with smaller cases of voice similarity analysis
and embedding extraction

The purpose is to ensure that the algorithms are working correctly and to test the
functionality of the code.
'''

import os
import torch
import torchaudio
import pickle
from transformers import Wav2Vec2FeatureExtractor
from transformers import WavLMForXVector
import numpy as np
import pandas as pd

def get_embeddings_for_wav(feature_extractor, model, accent, gender, speaker, wav_file):
    '''
    Load a WAV file, resample it to 16kHz if needed, and extract embeddings using WavLM.'''

    wav_path = os.path.join(
        os.path.dirname(__file__),
        'data',
        'cleansed',
        accent,
        gender,
        speaker,
        wav_file)

    if not os.path.exists(wav_path):
        raise FileNotFoundError(f'WAV file not found at: {wav_path}')

    waveform, original_sr = torchaudio.load(wav_path)

    # Resample to 16kHz if needed
    target_sr = 16000
    if original_sr != target_sr:
        resampler = torchaudio.transforms.Resample(orig_freq=original_sr, new_freq=target_sr)
        waveform = resampler(waveform)

    # Convert to 1D numpy array
    audio_array = waveform.squeeze().numpy()

    # Extract features
    inputs = feature_extractor(
        [audio_array],
        sampling_rate=target_sr,
        return_tensors='pt',
        padding=True)

    with torch.no_grad():
        embeddings = model(**inputs).embeddings
        embeddings = torch.nn.functional.normalize(embeddings, dim=-1)

    return embeddings

# import torch

def cosine_sim_two_wav(wav_1, wav_2):
    '''
    Compare two WAV files using WavLM and return the cosine similarity between their embeddings.
    '''

    local_model_folder = os.path.join(
        os.path.dirname(__file__),
        'model',
        'models--microsoft--wavlm-base-plus-sv',
        'snapshots',
        'feb593a6c23c1cc3d9510425c29b0a14d2b07b1e')

    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(local_model_folder)
    model = WavLMForXVector.from_pretrained(local_model_folder)

    e_1 = get_embeddings_for_wav(
        feature_extractor,
        model,
        wav_1['accent'],
        wav_1['gender'],
        wav_1['speaker'],
        wav_1['wav_file'])

    e_2 = get_embeddings_for_wav(
        feature_extractor,
        model,
        wav_2['accent'],
        wav_2['gender'],
        wav_2['speaker'],
        wav_2['wav_file'])

    # the resulting embeddings can be used for cosine similarity-based retrieval
    cosine_sim = torch.nn.CosineSimilarity(dim=-1)
    similarity = cosine_sim(e_1, e_2)

    return similarity

def create_embeddings_dict(wav_1, wav_2):
    '''
    Compare two WAV files using WavLM and return the cosine similarity between their embeddings.
    '''

    dict_embeddings = {}

    local_model_folder = os.path.join(
        os.path.dirname(__file__),
        'model',
        'models--microsoft--wavlm-base-plus-sv',
        'snapshots',
        'feb593a6c23c1cc3d9510425c29b0a14d2b07b1e')

    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(local_model_folder)
    model = WavLMForXVector.from_pretrained(local_model_folder)

    e_1 = get_embeddings_for_wav(
        feature_extractor,
        model,
        wav_1['accent'],
        wav_1['gender'],
        wav_1['speaker'],
        wav_1['wav_file'])

    speaker = wav_1['speaker']
    if speaker in dict_embeddings:
         dict_embeddings[speaker] = torch.cat([dict_embeddings[speaker],    e_1 ], dim=0)
    else:
        dict_embeddings[speaker] = e_1

    e_2 = get_embeddings_for_wav(
        feature_extractor,
        model,
        wav_2['accent'],
        wav_2['gender'],
        wav_2['speaker'],
        wav_2['wav_file'])

    speaker = wav_2['speaker']
    if speaker in dict_embeddings:
         dict_embeddings[speaker] = torch.cat([dict_embeddings[speaker],    e_2 ], dim=0)
    else:
        dict_embeddings[speaker] = e_2
        
    embedding_folder = os.path.join(os.path.dirname(__file__),
        'data',
        'embeddings')

    df = pd.DataFrame([
        {'Speaker': speaker, 'Embedding': emb}
        for speaker, emb in dict_embeddings.items()
    ])


    print(f'Embeddings shape: {df["Embedding"].shape}')
    print(f'Embedding 1 shape: {df["Embedding"][0][0].shape}')
    print(f'Embedding 2 shape: {df["Embedding"][0][1].shape}')

    with open(os.path.join(embedding_folder, 'embeddings_reduced.pkl'), 'wb') as f:
        pickle.dump(df, f)


def open_embeddings_dict():
    '''
    Open the embeddings dictionary from the pickle file.
    '''
    embedding_folder = os.path.join(os.path.dirname(__file__),
        'data',
        'embeddings')

    with open(os.path.join(embedding_folder, 'embeddings_reduced.pkl'), 'rb') as f:
        df = pickle.load(f)

    rec_1 = df.head(1)
    
    return df['Embedding'][0]




if __name__ == '__main__':

    wav_1_data = {
        'accent': 'brm_001',
        'gender': 'male',
        'speaker': 'ajh001',
        'wav_file': 'shortpassagea_CT.wav'
    }

    wav_2_data = {
        'accent': 'brm_001',
        'gender': 'male',
        'speaker': 'ajh001',
        'wav_file': 'shortpassageb_CT.wav'
    }

    sim = cosine_sim_two_wav(wav_1_data, wav_2_data)

    print(f'''Cosine similarity between {wav_1_data['speaker']} 
          using {wav_1_data['wav_file']} and {wav_2_data['speaker']} 
          using {wav_2_data['wav_file']}: {sim.item()}''')

    create_embeddings_dict(wav_1_data, wav_2_data)

    embeddings = open_embeddings_dict()

    print(f'Embeddings shape: {embeddings.shape}')
    print(f'Embedding 1 shape: {embeddings[0].shape}')
    print(f'Embedding 2 shape: {embeddings[1].shape}')


    cosine_sim = torch.nn.CosineSimilarity(dim=-1)
    sim = cosine_sim(embeddings[0], embeddings[1])
    
    print(f'Cosine similarity between embeddings from files:{sim.item()}')
