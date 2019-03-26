import os
import sys
from importlib import import_module
from clint.arguments import Args
from clint.textui import puts, colored, indent
from functools import partial
import numpy as np

import pdb

"""
Script that provides a simple user interface for exploring various versions of RELIEF based
algorithms.

Author: Jernej Vivod

"""


quit = False  # If true at end of script, it will not restart.

while True:
    ### User choice values ###

    # --- 
    usr_alg_choices = {1, 2, 3}  # Algorithm choices
    usr_alg_choice = None        # User's algorithm choice
    # ---
    usr_aug_choices = {1, 2, 3, 4}  # Algorithm augmentation choices
    usr_aug_choice = None           # User's augmentation choice
    # ---
    usr_metric_choices = {1, 2}     # Metric learning type choices
    usr_metric_choice = None        # User's metric learning type choice
    # --
    usr_dataset_choices = dict()  # Populated later
    usr_dataset_choice = None     # User's dataset choice
    # ---


    ### Parsing algorithm to use from user ###

    while True:
        os.system('clear')
        puts(colored.blue('Select algorithm to use:')) 
        with indent(4, quote=colored.blue('>>>')):
            puts(colored.green('Basic RELIEF (1)'))
            puts(colored.green('RELIEFF (2)'))
            puts(colored.green('Iterative RELIEF (3)'))

        alg_usr = input()
        if alg_usr.isdigit() and int(alg_usr) in usr_alg_choices:
            usr_alg_choice = int(alg_usr)
            break



    ### Parsing algorithm augmentation from user ###

    while True:
        os.system('clear')
        puts(colored.blue('Select algorithm augmentation:'))
        with indent(4, quote=colored.blue('>>>')):
            puts(colored.green('None (1)'))
            puts(colored.green('Metric learning (2)'))
            puts(colored.green('me-dissimilarity (3)'))
            puts(colored.green('mp-dissimilarity (4)'))

        aug_usr = input()
        if aug_usr.isdigit() and int(aug_usr) in usr_aug_choices:
            usr_aug_choice = int(aug_usr)
            break

    

    ### If user chose metric learning augmentation, ###
    ### parse type of metric learning to use from user ###
    
    if usr_aug_choice == 2:
        while True:
            os.system('clear')
            puts(colored.blue('Select type of metric learning to use:'))
            with indent(4, quote=colored.blue('>>>')):
                puts(colored.green('Covariance (1)'))
                puts(colored.green('PCA (2)'))

            metric_usr = input()
            if metric_usr.isdigit() and int(metric_usr) in usr_metric_choices:
                usr_metric_choice = int(metric_usr)
                break

    ### Parse dataset to use from user ###

    while True:
        os.system('clear')
        puts(colored.blue('Select dataset to use: '))
        with indent(4, quote=colored.blue('>>>')):
            for idx, dataset_name in enumerate(os.listdir(sys.path[0] + '/datasets')):
                usr_dataset_choices[idx+1] = dataset_name
                puts(colored.green('{0} ({1})'.format(dataset_name, idx+1)))
        dataset_usr = input()

        if dataset_usr.isdigit() and int(dataset_usr) in usr_dataset_choices.keys():
            usr_dataset_choice = int(dataset_usr)
            break


    ### Importing dataset ###

    import datasets.load_dataset as load_dataset
    data = load_dataset.load(usr_dataset_choices[usr_dataset_choice], 'data')
    target = load_dataset.load(usr_dataset_choices[usr_dataset_choice], 'target')




    ### 1. part of algorithm initialization  ###

    if usr_alg_choice == 1:
        from algorithms.relief import relief  # Partially pass parameters
        alg = partial(relief, data, target, data.shape[0], lambda x1, x2: np.sum(np.abs(x1-x2), 1))

        ### Handle augmentations ###
        # TODO
        # alg = partial(alg, ...)

    elif usr_alg_choice == 2:
        pass
    elif usr_alg_choice == 3:
        pass


    ## Augmented metric function initialization ##
    # TODO get dist func
    if usr_aug_choice == 1:
        pass
    elif usr_aug_choice == 2:
        if usr_metric_choice == 1:
            from augmentations.covariance import get_dist_func
            learned_metric = get_dist_func(data, target)
            alg = partial(alg, learned_metric_func=learned_metric)
        if usr_metric_choice == 2:
            pass



    ### Prompt user to press ENTER to start computations. ###
    os.system('clear')
    puts(colored.yellow('Press ENTER to start'))
    input()

    ### Call and time initialize algorithm ###
    rank, weights = alg()



    ### Display Results ###

    os.system('clear')
    puts(colored.yellow('Results:'))
    print('Feature ranks: {0}'.format(rank))
    print('Feature weights: {0}'.format(weights))
    print()


    ### Ask user if they want to restart the program ###

    while True:
        restart = input(colored.yellow('Run again? (y/n) '))
        if restart == 'y':
            break
        if restart == 'n':
            quit = True
            break
        else:
            pass

    if quit:
        break
    else:
        pass

