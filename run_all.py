from Classifier import *
from Knowledge import *
from Game import *

# Train the Classifier & scrape the Did-You-Know (DYK)
train_classifier()
scrape_Batch_of_DYKs()

# Run The Game:
run_Game()