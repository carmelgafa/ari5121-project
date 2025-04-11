'''
This script will go through all the speakers and calculate
the embeddings for all that files, in order to facilitate the 
comparison of the voices.
The data structure used to hold this information is a dictionary
with the following structure:
{str:list}
str: the name of the speaker
list: a list of lists, where each list contains the embeddings
for a single file.
The embeddings are stored in a list of floats.

Finally all the data is stored in a pandas dataframe and saved to a csv file.
'''

import os
import re
import torch
import torchaudio
import pandas as pd
from transformers import Wav2Vec2FeatureExtractor, WavLMForXVector

def create_embeddings_information():
    '''
    Goes through all the speakers and calculate
    the embeddings for all their files, in order to facilitate the 
    comparison of the voices.
    '''

    embeddings_dict = {}

    local_model_folder = os.path.join(
        os.path.dirname(__file__),
        'model',
        'models--microsoft--wavlm-base-plus-sv',
        'snapshots',
        'feb593a6c23c1cc3d9510425c29b0a14d2b07b1e')

    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(local_model_folder)
    model = WavLMForXVector.from_pretrained(local_model_folder)

    cleansed_folder = os.path.join(
        os.path.dirname(__file__),
        'data',
        'cleansed')

    # list accent folders removing the annoying folders
    accents = [accent_folder
               for accent_folder in os.listdir(cleansed_folder)]

    # go through all accents
    for accent in accents:
        accent_path = os.path.join(cleansed_folder, accent)

        # list all genders in each accent
        genders = [gender_folder
                   for gender_folder in os.listdir(accent_path)]

        # go through all genders in each accent
        for gender in genders:
            gender_folder = os.path.join(accent_path, gender)
            # go through each speaker in each gender
            # list all speakers

            speakers = [speaker
                        for speaker in os.listdir(gender_folder)]

            for speaker in speakers:
                speaker_path = os.path.join(gender_folder, speaker)

                for filename in os.listdir(speaker_path):
                    if re.fullmatch(r'shortpassage.*\.wav', filename):

                        print(accent, gender, speaker, filename)

                        waveform, original_sr = torchaudio.load(
                            os.path.join(
                                speaker_path,
                                filename))

                        # Resample to 16kHz if needed
                        target_sr = 16000
                        if original_sr != target_sr:
                            resampler = torchaudio.transforms.Resample(
                                orig_freq=original_sr,
                                new_freq=target_sr)
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

                        if speaker not in embeddings_dict:
                            embeddings_dict[speaker] = []

                        embeddings_dict[speaker].append(embeddings.squeeze().numpy())


    # Convert to a pandas df
    df = pd.DataFrame.from_dict(embeddings_dict, orient='index')

    embedding_folder = os.path.join(
        os.path.dirname(__file__),
        'data',
        'embeddings')

    if not os.path.exists(embedding_folder):
        os.makedirs(embedding_folder, exist_ok=True)

    # Save to a CSV file
    df.to_csv(os.path.join(embedding_folder, 'embeddings.csv'), index=True)



def check_embeddings_count():

    # read embeddings omitting index and headers

    df = pd.read_csv(os.path.join(
        os.path.dirname(__file__),
        'data',
        'embeddings',
        'embeddings.csv'),
        index_col=0,
        header=0)

    rec_1 = df.head(1)

    #print shape of element in second column of first row
    print('Number of embeddings:', rec_1.iloc[0][2].shape)







if __name__ == '__main__':
    # print('Calculating embeddings for all speakers...')
    # create_embeddings_information()

    check_embeddings_count()
