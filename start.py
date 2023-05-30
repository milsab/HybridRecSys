import torch
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR
import json
import yaml
import models
from run import Run


def run_experiment():
    # Set Model
    recsys = models.RecSysBinary(INPUT_SIZE, OUTPUT_SIZE, dropout=DROPOUT)

    # Set Criterion
    criterion = nn.BCELoss()

    # Set Optimizer
    optimizer = torch.optim.Adam(recsys.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    # Set LR_Scheduler
    lr_scheduler = StepLR(optimizer, step_size=LR_SCHEDULER_STEP, gamma=LR_SCHEDULER_GAMMA)

    run = Run(model=recsys, criterion=criterion, optimizer=optimizer, ratings_filename=RATING_FILE,
              users_embeddings_filename=USERS_EMBEDDINGS, items_embeddings_filename=ITEMS_EMBEDDINGS,
              tensorboard_name=NAME, wandb_name=NAME, select_data_size=SELECT_DATA_SIZE,
              batch_size=BATCH_SIZE, lr=LEARNING_RATE, epochs=EPOCHS, lr_scheduler=lr_scheduler,
              weight_decay=WEIGHT_DECAY,
              dropout=DROPOUT, training_ratio=TRAINING_RATIO, test_ratio=TEST_RATIO, data_name=DATA_NAME)

    print('\n\n***************  ' + NAME + '  ***************')
    run.start()


with open('config.yaml', 'r') as file:
    # Load the YAML file into a list of dictionaries
    documents = list(yaml.safe_load_all(file))

    EPOCHS = documents[0]['EPOCHS']
    SELECT_DATA_SIZE = documents[0]['SELECT_DATA_SIZE']
    BATCH_SIZE = documents[0]['BATCH_SIZE']
    INPUT_SIZE = documents[0]['INPUT_SIZE']
    OUTPUT_SIZE = documents[0]['OUTPUT_SIZE']

    LEARNING_RATE = documents[0]['LEARNING_RATE']
    LR_SCHEDULER_STEP = documents[0]['LR_SCHEDULER_STEP']
    LR_SCHEDULER_GAMMA = documents[0]['LR_SCHEDULER_GAMMA']
    WEIGHT_DECAY = documents[0]['WEIGHT_DECAY']
    DROPOUT = documents[0]['DROPOUT']

    TRAINING_RATIO = documents[0]['TRAINING_RATIO']
    TEST_RATIO = documents[0]['TEST_RATIO']

    for dataset in documents[1]:
        data = (dataset['data'])
        DATA_NAME = data['DATA_NAME']
        PATH = data['PATH']
        RATING_FILE = PATH + data['RATING_FILE']
        ITEMS_EMBEDDINGS = PATH + data['ITEMS_EMBEDDINGS']
        experiments = data['experiments']
        for experiment in experiments:
            NAME = experiment['NAME']
            USERS_EMBEDDINGS = PATH + experiment['USERS_EMBEDDINGS']
            run_experiment()
