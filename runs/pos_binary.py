import torch
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR

import models
from run import Run

INPUT_SIZE = 768
OUTPUT_SIZE = 1
BATCH_SIZE = 256
LEARNING_RATE = 0.001
EPOCHS = 40
SELECT_DATA_SIZE = 1024
WEIGHT_DECAY = 0.001
DROPOUT = False

ratings = 'ratings_5_10_binary.pkl'
items_embeddings = 'books_embeddings.pkl'
users_embeddings = 'users_embeddings_5_10_pos.pkl'
# users_embeddings = 'users_embeddings_5_10_pos_ignore4.pkl'
# users_embeddings = 'users_embeddings_5_10_pos_zscore.pkl'

tensorboard_name = 'pos_binary'
wandb_name = 'pos_binary'

# Set Model
recsys = models.RecSysBinary(INPUT_SIZE, OUTPUT_SIZE, dropout=DROPOUT)

# Set Criterion
criterion = nn.BCELoss()

# Set Optimizer
optimizer = torch.optim.Adam(recsys.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

# Set LR_Scheduler
lr_scheduler = StepLR(optimizer, step_size=20, gamma=0.1)

pos_binary_run = Run(model=recsys, criterion=criterion, optimizer=optimizer, ratings_filename=ratings,
                     users_embeddings_filename=users_embeddings, items_embeddings_filename=items_embeddings,
                     tensorboard_name=tensorboard_name, wandb_name=wandb_name, select_data_size=SELECT_DATA_SIZE,
                     batch_size=BATCH_SIZE, lr=LEARNING_RATE, epochs=EPOCHS, lr_scheduler=lr_scheduler,
                     weight_decay=WEIGHT_DECAY,
                     dropout=DROPOUT)

print('***** POSITIVE BINARY EXPERIMENT *****')
pos_binary_run.start()
