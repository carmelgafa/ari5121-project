'''
Execute the voice similarity analysis pipeline
1. download of model
2. download of data
3. clean up of data
4. calculation of embeddings
5. evaluation of similarity
'''


from data_download import create_folders, download_data, download_model
from data_cleanse import  cleanse_data
from calc_population_embeddings import generate_embeddings_information
from calc_embeddings_comparison import compare_embeddings, expand_summary_table
from utils import log_message

def execute_pipeline():
    '''
    Execute the whole pipeline
    '''
    create_folders()
    download_data()
    download_model()
    cleanse_data()
    generate_embeddings_information()
    compare_embeddings()
    expand_summary_table()



if __name__ == '__main__':

    log_message('Starting pipeline...')
    execute_pipeline()
    log_message('Pipeline completed.')
