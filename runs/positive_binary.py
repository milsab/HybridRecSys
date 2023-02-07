import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import StepLR, ExponentialLR
import shutil

from goodreads import preprocessing
from goodreads import models
from goodreads import data
from goodreads import experiment

PATH = '../../datasets/goodreads/comics/'

# Set Tensorboard
shutil.rmtree('tensorboard/positive_binary', ignore_errors=True)
writer = SummaryWriter('tensorboard/positive_binary')

# Detect Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('DEVICE IS: ', device)

# Hyper Parameters
EPOCHS = 100
BATCH_SIZE = 4
INPUT_SIZE = 768
OUTPUT_SIZE = 1
LEARNING_RATE = 0.001


# Create Model
# recsys = models.RecSys(INPUT_SIZE, OUTPUT_SIZE)
recsys = models.RecSysBinary(INPUT_SIZE, OUTPUT_SIZE)
recsys.to(device)

# Load Ratings
ratings = preprocessing.load_data(path=PATH, filename='ratings_5_10_binary.pkl')
ratings = ratings[:1000]

# Split Data
x_train, y_train, x_val, y_val, x_test, y_test = preprocessing.split_data(ratings, 0.9, 0.1)

# Load Embeddings
users_embeddings = preprocessing.load_data(path=PATH, filename='users_embeddings_5_10_pos.pkl')
items_embeddings = preprocessing.load_data(path=PATH, filename='books_embeddings.pkl')

# Create DataLoaders
train_ds = data.GoodreadsDataset(x_train, y_train, users_embeddings, items_embeddings)
train_loader = DataLoader(dataset=train_ds, batch_size=BATCH_SIZE)

val_ds = data.GoodreadsDataset(x_val, y_val, users_embeddings, items_embeddings)
val_loader = DataLoader(dataset=val_ds, batch_size=BATCH_SIZE)

test_ds = data.GoodreadsDataset(x_test, y_test, users_embeddings, items_embeddings)
test_loader = DataLoader(dataset=test_ds, batch_size=BATCH_SIZE)

# Set Criterion
criterion = nn.BCELoss()

# Set Optimizer
optimizer = torch.optim.Adam(recsys.parameters(), lr=LEARNING_RATE)

# Set LR_Scheduler
exp_lr_scheduler = StepLR(optimizer, step_size=50, gamma=0.1)
# exp_lr_scheduler = ExponentialLR(optimizer, gamma=0.1)


# Execute Experiment
experiment = experiment.Experiment(recsys, train_loader, val_loader, test_loader,
                                   criterion, optimizer, device, writer, EPOCHS, LEARNING_RATE)

experiment.run(exp_lr_scheduler)
