'''
Utility functions
'''


import time
import os

def timer(func):
    '''
    A decorator to time the execution of a function
    '''
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f"{func.__name__} executed in {(end - start):.4f} seconds")


        log_message(f"{func.__name__} executed in {(end - start):.4f} seconds")

        return result
    return wrapper

def log_message(message):
    '''
    Log a message to the log file
    '''
    # folder existence check
    log_folder = os.path.join(os.path.dirname(__file__), 'logs')
    if not os.path.exists(log_folder):
        os.makedirs(log_folder, exist_ok=True)

    # file existence check
    log_file = os.path.join(log_folder, 'log.txt')
    if not os.path.exists(log_file):
        with open(log_file, 'w', encoding='utf-8') as f:
            f.write('Log file created\n')

    log_string = f"{time.strftime('%Y-%m-%d %H:%M:%S')} - {message}\n" #pylint: disable=line-too-long

    # append
    with open(log_file, 'a', encoding='utf-8') as f:
        f.write(log_string)
