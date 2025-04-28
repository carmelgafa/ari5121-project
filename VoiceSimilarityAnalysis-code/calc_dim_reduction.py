''' 
processing the embeddings and comparing them
1. compare the embeddings
2. expand the summary table by splitting the speaker
'''


import os
import pandas as pd
import torch
from utils import timer


@timer
def compare_pca(attempt_gpu:bool=False):
    '''
    
    '''

    device = torch.device("cuda" if attempt_gpu and torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # create results folder in data
    results_folder = os.path.join(
        os.path.dirname(__file__),
        'data',
        'results')
    if not os.path.exists(results_folder):
        os.makedirs(results_folder, exist_ok=True)

    df_speakers = pd.read_pickle(os.path.join(
        os.path.dirname(__file__),
        'data',
        'embeddings',
        'embeddings.pkl'))


    use pca to reduce to 2 dimensions and expand speaker from 'acce'

    df_results_raw = pd.DataFrame(columns=['Speaker_1', 'Speaker_2', 'Cosine_Similarity'])
    df_results_summary = pd.DataFrame(columns=['Speaker_1', 'Speaker_2', 'Cosine_Similarity'])

    # load cosine sim
    cosine_sim = torch.nn.CosineSimilarity(dim=-1)

    # go through all speakers
    for i in range(len(df_speakers)):
        # get the speaker name
        speaker_1 = df_speakers['Speaker'][i]

        # get the embedding
        speaker_1_embeddings = df_speakers['Embedding'][i]

        print(f'self similarity for {speaker_1}')

        min_self_similarity = 1.0

        for k, embedding_1 in enumerate(speaker_1_embeddings):
            for _, embedding_2 in enumerate(speaker_1_embeddings[k+1:]):
                similarity = cosine_sim(embedding_1.to(device), embedding_2.to(device))

                df_results_raw = pd.concat([
                    df_results_raw,
                        pd.DataFrame([{
                            'Speaker_1': speaker_1,
                            'Speaker_2': speaker_1,
                            'Cosine_Similarity': similarity.item()
                        }])
                    ], ignore_index=True)

                min_self_similarity = min(min_self_similarity, similarity.item())

        df_results_summary = pd.concat([
                df_results_summary,
                pd.DataFrame([{
                    'Speaker_1': speaker_1,
                    'Speaker_2': speaker_1,
                    'Cosine_Similarity': min_self_similarity
                }])
            ], ignore_index=True)

        for j in range(i + 1, len(df_speakers)):
            # get the speaker name
            speaker_2 = df_speakers['Speaker'][j]

            # get the embedding for the speaker
            speaker_2_embeddings = df_speakers['Embedding'][j]

            print(f'Comparing {speaker_1} with {speaker_2}')

            max_similarity = 0.0

            for embedding_1 in speaker_1_embeddings:
                for embedding_2 in speaker_2_embeddings:
                    similarity = cosine_sim(embedding_1, embedding_2)

                    df_results_raw = pd.concat([
                        df_results_raw,
                            pd.DataFrame([{
                                'Speaker_1': speaker_1,
                                'Speaker_2': speaker_2,
                                'Cosine_Similarity': similarity.item()
                            }])
                        ], ignore_index=True)

                    max_similarity = max(max_similarity, similarity.item())

            df_results_summary = pd.concat([
                df_results_summary,
                    pd.DataFrame([{
                        'Speaker_1': speaker_1,
                        'Speaker_2': speaker_2, 
                        'Cosine_Similarity': max_similarity}])
                    ], ignore_index=True)

    df_results_raw.to_csv(os.path.join(
        os.path.dirname(__file__),
        'data',
        'results',
        'similarity_results_raw.csv'))

    df_results_summary.to_csv(os.path.join(
        os.path.dirname(__file__),
        'data',
        'results',
        'similarity_results_summary.csv'))


@timer
def expand_summary_table():
    # Load your data
    '''
    Expands the summary table by splitting the speaker name into
    accent, gender and speaker. The resulting table is then saved
    to a csv file.
    '''

    df_summary = pd.read_csv(os.path.join(
        os.path.dirname(__file__),
        'data',
        'results',
        'similarity_results_summary.csv'))

    df_summary_processed = pd.DataFrame(
        columns=[
            '1_accent',
            '1_gender',
            '1_speaker',
            '2_accent',
            '2_gender',
            '2_speaker',
            'Cosine_Similarity'])

    # go through all rows
    for _, row in df_summary.iterrows():

        accent_1 = row['Speaker_1'].split('-')[0]
        gender_1 = row['Speaker_1'].split('-')[1]
        speaker_1 = row['Speaker_1'].split('-')[2]

        accent_2 = row['Speaker_2'].split('-')[0]
        gender_2 = row['Speaker_2'].split('-')[1]
        speaker_2 = row['Speaker_2'].split('-')[2]

        cosine_similarity = row['Cosine_Similarity']

        # add
        df_summary_processed = pd.concat([df_summary_processed, pd.DataFrame([{
                '1_accent': accent_1,
                '1_gender': gender_1,
                '1_speaker': speaker_1,
                '2_accent': accent_2,
                '2_gender': gender_2,
                '2_speaker': speaker_2,
                'Cosine_Similarity': cosine_similarity
            }])], ignore_index=True)

    # Save
    df_summary_processed.to_csv(os.path.join(
        os.path.dirname(__file__),
        'data',
        'results',
        'similarity_results_summary_processed.csv'), index=False)


if __name__ == '__main__':
    compare_embeddings()

    expand_summary_table()
