'''
Created on 18. 3. 2014

@author: Vancikv
'''

import os

def get_outfile(folder_name, file_name):
    '''Returns a file in the specified folder using the home
    directory as root.
    '''
    HOME_DIR = os.path.expanduser("~")
    out_dir = os.path.join(HOME_DIR, folder_name)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    outfile = os.path.join(out_dir, file_name)
    return outfile
