import os
import shutil
from datetime import datetime
import csv

def start_sys(paths,if_scratch=True):
    critic_path  = paths['critic_path']
    policy_path  = paths['policy_path']
    rp_path      = paths['rp_path']
    log_path     = paths['log_path']
    
    # Remove olders if exist
    remove_folders(critic_path)
    remove_folders(policy_path)
    remove_folders(log_path)
    remove_folders(rp_path)

    create_folders(critic_path)
    create_folders(policy_path)
    create_folders(log_path)
    create_folders(rp_path)


def create_folders(path):
    if (os.path.exists(path) and os.path.isdir(path) ) == False :
        os.mkdir(path)    

def remove_folders(path):
    if os.path.exists(path) and os.path.isdir(path):
        shutil.rmtree(path)

def construct_str( dict_name, var_name,is_date=False):
    name = dict_name + str(var_name) + "\n"
    if is_date : name = name + "\n"
    return name

# Writes All Hyperparameters into the Log File
def log_hyperparams( hyperparameters_dict,log_file_name ,reward_file_name):
    # Make a One Big String to Write into the Log File Later 
    params_str =   construct_str( "Log Date :   ", datetime.now(),True)
    for key,value in hyperparameters_dict.items()  :
        params_str += construct_str(key+ " :  ",value)

    # Write All Info to the Log File
    log_file_name = log_file_name + '.txt'
    log_txt = open(log_file_name ,'w')
    with open(log_file_name, 'a') as out:
        out.write(params_str + '\n')

    reward_file_name = reward_file_name + '.csv'
    reward_Csv = open(reward_file_name, 'w')