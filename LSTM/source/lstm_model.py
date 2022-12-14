import torch
import torch.nn as nn
import torch.nn.functional as F

class causalmodel(nn.ModuleList):

	def __init__(self, args):
		super(causalmodel, self).__init__()
		
		# Initilasing the paarmeters used in the LSTM Network
		self.batch_size = args.batch_size
		self.hidden_dim = args.hidden_dim
		self.LSTM_layers = args.lstm_layers
		self.input_size = args.max_words # embedding dimention
		
		# Adding droput layer
		self.dropout = nn.Dropout(0.5)
		self.embedding = nn.Embedding(self.input_size, self.hidden_dim, padding_idx=0)
		# Main LSTM Layer added with argument pased parameters for the model
		self.lstm = nn.LSTM(input_size=self.hidden_dim, hidden_size=self.hidden_dim, num_layers=self.LSTM_layers, batch_first=True)
		# Two fully connected layers added to output a 1d vector finally
		self.fc1 = nn.Linear(in_features=self.hidden_dim, out_features=256)
		self.fc2 = nn.Linear(256, 1)
		

	def forward(self, x):
	    
		# Intilaising with zeros for hidden and cell states
		h = torch.zeros((self.LSTM_layers, x.size(0), self.hidden_dim))
		c = torch.zeros((self.LSTM_layers, x.size(0), self.hidden_dim))
		
		# FasText embedding of the input is considered
		# Tokenized word embeddings input is passed to LSTM
		out, (hidden, cell) = self.lstm(out, (h,c))
		# Drop out layer is added
		out = self.dropout(out)
		# Relu activaion layer is added 
		out = torch.relu_(self.fc1(out[:,-1,:]))
		# Drop out layer along with sigmoid activation
		out = self.dropout(out)
		out = torch.sigmoid(self.fc2(out))

		return out