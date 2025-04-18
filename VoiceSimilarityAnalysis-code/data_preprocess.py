'''
This script preprocesses the ABI-1 Corpus dataset implementing resampling
'''


import os
import re
import torchaudio
import numpy as np
from utils import timer

@timer
def preprocess_data():
    '''
    This function preprocesses the ABI-1 Corpus dataset implementing resampling
    '''

    cleansed_accents_folder = os.path.join(
        os.path.dirname(__file__),
        'data',
        'cleansed')

    preprocessed_folder = os.path.join(
        os.path.dirname(__file__),
        'data',
        'preprocessed')

    if not os.path.exists(preprocessed_folder):
        os.makedirs(preprocessed_folder, exist_ok=True)

    # list accent folders removing the annoying folders
    accents = [accent_folder
               for accent_folder in os.listdir(cleansed_accents_folder)
               if not accent_folder.startswith('.')]

    # go through all accents
    for accent in accents:
        accent_path = os.path.join(cleansed_accents_folder, accent)

        # list all genders in each accent
        genders = [gender_folder
                   for gender_folder in os.listdir(accent_path)
                   if not gender_folder.startswith('.')]

        # go through all genders in each accent
        for gender in genders:
            gender_folder = os.path.join(accent_path, gender)
            # go through each speaker in each gender
            # list all speakers

            speakers = [speaker
                        for speaker in os.listdir(gender_folder)
                        if not speaker.startswith('.')]

            for speaker in speakers:
                speaker_path = os.path.join(gender_folder, speaker)

                dest_path = os.path.join(preprocessed_folder, accent, gender, speaker)
                os.makedirs(dest_path, exist_ok=True)

                # copy only filenames that  are shortpassage*.wav
                # go through all files
                for filename in os.listdir(speaker_path):
                    if re.fullmatch(r'shortpassage.*\.wav', filename):

                        waveform, original_sr = torchaudio.load(
                            os.path.join(speaker_path, filename))

                        target_sr = 16000
                        if original_sr != target_sr:
                            resampler = torchaudio.transforms.Resample(
                                orig_freq=original_sr, new_freq=target_sr)
                            waveform = resampler(waveform)

                        audio_array = waveform.squeeze().numpy()

                        # save the resampled numpy array
                        np.save(os.path.join(dest_path, filename), audio_array)


if __name__ == '__main__':
    print('Preprocessing ABI-1 Corpus dataset...')
    preprocess_data()
    print('Preprocessing completed.')
