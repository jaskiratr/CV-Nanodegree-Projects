import torch
import torch.nn as nn
import torchvision.models as models


class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        # Resnets: https://github.com/KaimingHe/deep-residual-networks
        resnet = models.resnet152(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        '''
        Take all the layers of the resnet152 model, except the last one that outputs classes. 
        We'll use the feature vectors, second last layer as context for the DecoderRNN.
        '''
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)
        self.bn = nn.BatchNorm1d(embed_size, momentum=0.01)

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        features = self.bn(features)
        return features
    

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        # Initialize the decoder
        super(DecoderRNN, self).__init__()
        
        # From the notebook
        self.batch_size = 64
        
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.num_layers = num_layers 
        self.embed = nn.Embedding(vocab_size, embed_size)
        # LSTM Cell
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)
        
    def forward(self, features, captions):
        '''
        Forward Pass:
        Inputs are input = features and hidden cell state = captions?
        '''
        embed_size = self.embed_size
        hidden_size = self.hidden_size
        vocab_size = self.vocab_size
        num_layers = self.num_layers
        
        # Trimming the end last word so that outputs.shape[1]==captions.shape[1]. 
        ## Otherwise it increments by 1 // https://knowledge.udacity.com/questions/2866
        embeddings = self.embed(captions[:, :-1])
        inputs = torch.cat((features.unsqueeze(1), embeddings), 1)
        hiddens, _ = self.lstm(inputs)
        outputs = self.linear(hiddens)
        
        return outputs

    def sample(self, inputs, states=None, max_len=20):
        '''
        Accepts pre-processed image tensor (inputs)
        Returns predicted sentence (list of tensor ids of length max_len) 
        '''
        embed_size = self.embed_size
        hidden_size = self.hidden_size
        vocab_size = self.vocab_size
        num_layers = self.num_layers
        
        # Predicted sentence
        sampled_ids = []
        features = inputs
        # Generate a predicted word per LSTM cell.
        for i in range(max_len):
            hiddens, states = self.lstm(features, states)
            outputs = self.linear(hiddens.squeeze(1))
            _ , predicted = torch.max(outputs,1)
            sampled_ids.append(predicted)
            features = self.embed(predicted)
            features = features.unsqueeze(1)
        
        # Stack the LSTM cells output
        sampled_ids = torch.stack(sampled_ids, 1)
        return sampled_ids