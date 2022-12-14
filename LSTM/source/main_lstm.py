
import numpy as np
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import logging
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from main_lstm import causalmodel
from preprocessing_lstm import Preprocessing
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchtext.vocab import FastText
import io

logger = logging.getLogger(__name__)


# Used for appropriate Data preprocesing in LSTM's
class DatasetMaper(Dataset):
	def __init__(self, x, y):
		self.x = x
		self.y = y

	def __len__(self):
		return len(self.x)

	def __getitem__(self, idx):
		return self.x[idx], self.y[idx]


# As arguments are given through command line, special execute class is defined
class Execute:

	def __init__(self, args):
		## intialising the arguments
		self.__init_data__(args)
		self.args = args
		self.batch_size = args.batch_size
		# Intiliasing the model with arguments provided
		self.model = causalmodel(args)

	def __init_data__(self, args):

		# Initialize preprocessing from raw dataset to dataset split into training and testing
		# Training and test datasets are index strings that refer to tokens

		self.preprocessing = Preprocessing(args)
		self.preprocessing.load_data()
		self.preprocessing.prepare_tokens()

		raw_x_train = self.preprocessing.x_train
		raw_x_test = self.preprocessing.x_test

		self.y_train = self.preprocessing.y_train
		self.y_test = self.preprocessing.y_test
        
		# Tokensing the inputs to pass to the LSTM layers
		self.x_train = self.preprocessing.sequence_to_token(raw_x_train)
		self.x_test = self.preprocessing.sequence_to_token(raw_x_test)

	def train(self,word2vec):
		logger.info("***** train metrics *****")

		## Mapping the dataset

		training_set = DatasetMaper(self.x_train, self.y_train)
		test_set = DatasetMaper(self.x_test, self.y_test)

		self.loader_training = DataLoader(training_set, batch_size=self.batch_size)
		self.loader_test = DataLoader(test_set)


		optimizer = optim.RMSprop(self.model.parameters(), lr=args.learning_rate)
		## Now performing the training on data and evaluating the model
		for epoch in range(args.epochs):

			## to store the predictions

			predictions = []


			for x_batch, y_batch in self.loader_training:

				x = x_batch.type(torch.LongTensor)
				y = y_batch.type(torch.FloatTensor)

				y_pred = self.model(x)

				## calculating the loss

				loss = F.binary_cross_entropy(y_pred, y)

				optimizer.zero_grad()

				##backpropagation

				loss.backward()

				optimizer.step()

				predictions += list(y_pred.squeeze().detach().numpy())

			test_predictions = self.evaluation()

			## calculating the accuracy of test and train

			train_accuary = self.calculate_accuray(self.y_train, predictions)
			test_accuracy = self.calculate_accuray(self.y_test, test_predictions)

			print("Epoch: %d, loss: %.5f, Train accuracy: %.5f, Test accuracy: %.5f" %
			      (epoch+1, loss.item(), train_accuary, test_accuracy))

	def evaluation(self):
		logger.info("***** eval metrics *****")

		predictions = []
		self.model.eval()
		with torch.no_grad():
			for x_batch, y_batch in self.loader_test:
				x = x_batch.type(torch.LongTensor)
				y = y_batch.type(torch.FloatTensor)

				## preidicting the results

				y_pred = self.model(x)
				predictions += list(y_pred.detach().numpy())

		return predictions

	
	def calculate_accuray(grand_truth, predictions):
		logger.info("***** Predict *****")
		true_positives = 0
		true_negatives = 0

		for true, pred in zip(grand_truth, predictions):

			## calulating the accuracy by analying the predictions
			if (pred > 0.5) and (true == 1):
				true_positives += 1
			elif (pred < 0.5) and (true == 0):
				true_negatives += 1
			else:
				pass

		return (true_positives+true_negatives) / len(grand_truth)






## arguments passing

parser = argparse.ArgumentParser(description="Causal News corpus")

parser.add_argument("--epochs",dest="epochs",type=int,default=10,help="Number of gradient descent iterations. Default is 200.")

parser.add_argument("--learning_rate",dest="learning_rate",type=float,default=0.01,help="Gradient descent learning rate. Default is 0.01.")

parser.add_argument("--hidden_dim",dest="hidden_dim",type=int,default=128,help="Number of neurons by hidden layer. Default is 128.")

parser.add_argument("--lstm_layers",dest="lstm_layers",type=int,default=2,help="Number of LSTM layers")

parser.add_argument("--batch_size",dest="batch_size",type=int,default=64,help="Batch size")

parser.add_argument("--test_size",dest="test_size",type=float,default=0.20,help="Size of test dataset. Default is 10%.")

parser.add_argument("--max_len",dest="max_len",type=int,default=20,help="Maximum sequence length per tweet")

parser.add_argument("--max_words",dest="max_words",type=float,default=1000,help="Maximum number of words in the dictionary")

args = parser.parse_args()



execute = Execute(args)
execute.train()
execute.evaluation(Dataset)
logger.info("***** Causal News Corpus(Team : Thunderbolts) *****")