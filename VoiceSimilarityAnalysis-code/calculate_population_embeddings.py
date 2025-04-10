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
'''

import os
import re
import shutil

def create_embeddings_information():
    '''
    '''

    raw_accents_folder = os.path.join(
        os.path.dirname(__file__),
        'data',
        'raw',
        'ABI-1_Corpus',
        'ABI-1 Corpus',
        'accents')

    cleansed_folder = os.path.join(
        os.path.dirname(__file__),
        'data',
        'cleansed')

    # list accent folders removing the annoying folders
    accents = [accent_folder
               for accent_folder in os.listdir(raw_accents_folder)
               if not accent_folder.startswith('.')]

    # go through all accents
    for accent in accents:
        accent_path = os.path.join(raw_accents_folder, accent)

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

                # store resulting  data in cleaned/accent/gender/speaker
                dest_path = os.path.join(cleansed_folder, accent, gender, speaker)
                os.makedirs(dest_path, exist_ok=True)

                # copy only filenames that  are shortpassage*.wav
                # go through all files
                for filename in os.listdir(speaker_path):
                    if re.fullmatch(r'shortpassage.*\.wav', filename):

                        src_file = os.path.join(speaker_path, filename)
                        dst_file = os.path.join(dest_path, filename)
                        shutil.copy2(src_file, dst_file)


if __name__ == '__main__':
    print('Cleansing ABI-1 Corpus dataset...')
    create_embeddings_information()
    print('Cleansing completed.')
