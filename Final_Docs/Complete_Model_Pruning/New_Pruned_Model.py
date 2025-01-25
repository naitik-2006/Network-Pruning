# -*- coding: utf-8 -*-
"""
Created on Wed Jun 12 10:02:42 2024

@author: jishu
"""

import torch
import torch.nn as nn
import math
from torch import Tensor

import torchvision.models as models

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# For the encoder CNN we just need to call the pre-trained resnet fine tuned on our task



class EncoderCNN(nn.Module):
    def __init__(self , embedding_size = 512 , train_CNN = False):
        super(EncoderCNN , self).__init__()
        self.embedding_size = embedding_size 
        self.train_CNN = train_CNN
        resnet = models.resnet101(pretrained=True)
        
        # Remove the classification layer (avgpool and fc) at the end
        modules = list(resnet.children())[:-2]
        self.resnet = nn.Sequential(*modules) # Size (B x 2048 x H x W)

        
        # Fully connected layer to transform features to embedding size
        self.fc = nn.Linear(resnet.fc.in_features, embedding_size)
        
        # We do not freeze the parameters of the resnet here because we have to prune the model
        if not train_CNN:
            for params in self.resnet.parameters():
                params.requires_grad = False

    def forward(self, images):
        features = self.resnet(images)
        features = features.reshape(features.shape[0] , features.shape[1] , -1)
        features = features.permute(2 , 0 , 1)
        features = self.fc(features)
        return features
   
class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

        
    def forward(self, x: Tensor) -> Tensor:
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        #print(self.pe.shape)
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class TransformersEncoder(nn.Module):
    def __init__(self, embed_size, num_heads, num_encoder_layers, dropout):
        super(TransformersEncoder, self).__init__()
        self.num_heads = num_heads
        self.embed_size = embed_size
        self.num_encoder_layers = num_encoder_layers
        self.dropout = dropout

        self.encoder_layer = nn.TransformerEncoderLayer(d_model=self.embed_size, nhead=self.num_heads)
        self.encoder = nn.TransformerEncoder(encoder_layer=self.encoder_layer, num_layers=self.num_encoder_layers)

        self.positional_encoding = PositionalEncoding(d_model=embed_size)

    def forward(self, x): # Images are in the shape of (batch_size x embedding_dim)
        
        #encoder_output = x.permute(1 , 0 , 2) # (seq_len x batch_size x embeding_dim)
        encoder_output = self.positional_encoding(x) # (1 x batch_size x embedding_dim)
        encoder_output = self.encoder(encoder_output)
        
        return encoder_output
    
    def get_encoder_weights(self):
        weights = []
        
        for name , param in self.encoder.named_parameters():
            if 'weight' in name:
                weights.append(param)
            
        return weights
    
    def compute_penalty(self , lambda_reg = 0.1):
        weights = self.get_encoder_weights()
        
        loss = torch.tensor(0.).to(device)
        for w in weights:
            loss += torch.sum(w ** 2) # This is the L2 Regularization
            
        return (loss * lambda_reg)
            
    
# Now we make the Transformer decoder
class CustomTransformerDecoder(nn.TransformerDecoder):
    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None, tgt_key_padding_mask=None, memory_key_padding_mask=None):
        output = tgt
        hidden_states = []
        for mod in self.layers:  # Iterate over each decoder layer
            output = mod(output, memory, tgt_mask=tgt_mask, memory_mask=memory_mask,
                         tgt_key_padding_mask=tgt_key_padding_mask, memory_key_padding_mask=memory_key_padding_mask)
            hidden_states.append(output)
        if self.norm is not None:
            output = self.norm(output)
        return output, hidden_states

class TransformersDecoder(nn.Module):
    def __init__(self, embedding_size, trg_vocab_size, num_heads, num_decoder_layers, dropout = 0.2):
        super(TransformersDecoder, self).__init__()
        
        self.num_heads = num_heads
        self.embedding = nn.Embedding(trg_vocab_size, embedding_size)
        self.pos = PositionalEncoding(d_model=embedding_size)
        self.decoder_layer = nn.TransformerDecoderLayer(d_model=embedding_size, nhead=num_heads)
        self.decoder = CustomTransformerDecoder(self.decoder_layer, num_layers=num_decoder_layers)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.linear = nn.Linear(embedding_size, trg_vocab_size)
        self.drop = nn.Dropout(dropout)
        self.padding_idx = 2
        
    def make_mask(self, sz):
        mask = torch.zeros((sz, sz), dtype=torch.float32)
        for i in range(sz):
            for j in range(sz):
                if j > i: mask[i][j] = float('-inf')
        return mask
    
    def forward(self, features, caption):
        tgt_seq_length, N = caption.shape
        
        embed = self.drop(self.embedding(caption))
        embed = self.pos(embed)
        
        # Make the causal mask
        trg_mask = self.make_mask(tgt_seq_length).to(self.device)
        # Make the padding mask
        padding_mask = (caption == self.padding_idx)
        padding_mask = padding_mask.transpose(0,1).to(self.device)
        
        output, hidden_states = self.decoder(tgt=embed, memory=features, tgt_mask=trg_mask , tgt_key_padding_mask = padding_mask)
        
        output = self.linear(output)
        
        return output, hidden_states
    
class DecoderPruning(nn.Module):
    def __init__(self, decoder_network, num_selected_layers , trg_vocab_size , embedding_size , dropout = 0.2):
        super(DecoderPruning, self).__init__()
        self.num_selected_layers = num_selected_layers
        self.selected_layers = self.select_layers(decoder_network, num_selected_layers)
        self.embedding = nn.Embedding(trg_vocab_size, embedding_size)
        self.pos = PositionalEncoding(d_model=embedding_size)
        self.linear = nn.Linear(embedding_size, trg_vocab_size)
        self.drop = nn.Dropout(dropout)
        self.padding_idx = 2
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        
    def select_layers(self, decoder_network, num_selected_layers):
        num_total_layers = len(decoder_network.layers)
        selected_layers_indices = [0, num_total_layers - 1]  # Ensure the first and last layers are included

        # Determine the layers to include in the sub-network
        for i in range(1, num_selected_layers - 1):
            idx = int((num_total_layers - 1) * i / (num_selected_layers - 1))
            selected_layers_indices.append(idx)
    
        selected_layers_indices = list(sorted(set(selected_layers_indices)))  # Ensure unique and sorted indices
        selected_layers = nn.ModuleList([decoder_network.layers[idx] for idx in selected_layers_indices])
    
        return selected_layers
    
    def make_mask(self, sz):
        mask = torch.zeros((sz, sz), dtype=torch.float32)
        for i in range(sz):
            for j in range(sz):
                if j > i: mask[i][j] = float('-inf')
        return mask
    
    def forward(self, Input, memory):
        hidden_states = []
        tgt_seq_length ,  N  = Input.shape
        
        # Now make the target mask
        tgt_mask = self.make_mask(tgt_seq_length).to(self.device)
        # We also have to make the padding mask
        padding_mask = (Input == self.padding_idx)
        padding_mask = padding_mask.transpose(0,1).to(self.device)
        # Convert the input with proper embeddings and masks
        
        embed = self.drop(self.embedding(Input))
        embed = self.pos(embed)
         
       
        for layer in self.selected_layers:
            embed = layer(embed, memory, tgt_mask=tgt_mask,
                          tgt_key_padding_mask=padding_mask)
            hidden_states.append(embed)
        output = self.linear(embed)
        
        return output, hidden_states
        
class EncodertoDecoder(nn.Module):
    def __init__(self,embeding_size=512,trg_vocab_size=2994,num_heads=8,num_decoder_layers=4,dropout=0.02 , pruned_resnet_model_path = None):
        super(EncodertoDecoder,self).__init__()
  
        #self.image_encoder = EncoderCNN(embeding_size)
        
        self.encoder = TransformersEncoder(embeding_size, num_heads, 4 , dropout)
        
        self.decoder = TransformersDecoder(embeding_size, trg_vocab_size, num_heads, num_decoder_layers, dropout)
        
       
        self.pruned_image_encoder = torch.load(pruned_resnet_model_path)
                                           
        
    def forward(self , image , caption):
        
        
       # features = self.image_encoder(image) #This one is for without pruning
        features = self.pruned_image_encoder(image) # This one is for pruned_model
       
        features = self.encoder(features)
        
        
        output , hidden_states = self.decoder(features , caption)
        
        return output , hidden_states
    
class PrunedEncodertoDecoder(nn.Module):
    def __init__(self , embeding_size = 512 , trg_vocab_size = 2994 , num_heads = 8 , num_decoder_layers = 4 , dropout = 0.2 , num_selected_decoder_layers = 3, pruned_resnet_model_path = None):
        super(PrunedEncodertoDecoder , self).__init__()
        
        self.pruned_image_encoder = torch.load(pruned_resnet_model_path)
        self.encoder = TransformersEncoder(embeding_size , num_heads , 4 , dropout)
        self.original_decoder = TransformersDecoder(embeding_size, trg_vocab_size, num_heads, num_decoder_layers, dropout)
        self.pruned_decoder = DecoderPruning(self.original_decoder.decoder, num_selected_layers = num_selected_decoder_layers , trg_vocab_size=trg_vocab_size , embedding_size=embeding_size)
        
    def forward(self , image , caption):
        
        features = self.pruned_image_encoder(image)
        print(features.shape)
        features = self.encoder(features)
        print(features.shape)
        output , hidden_states = self.pruned_decoder(caption , features)
        
        return output , hidden_states
        
    
    
    
    
    
    
    