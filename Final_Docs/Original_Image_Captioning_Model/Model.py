import torch
from torch import Tensor
import math
import torchvision.models as models
import torch.nn as nn

class EncoderCNN(nn.Module):
    def __init__(self, embedding_size=512, train_CNN=False):
        super(EncoderCNN, self).__init__()
        self.embedding_size = embedding_size
        self.train_CNN = train_CNN
        
        
        # Load pre-trained ResNet101 model
        resnet = models.resnet101(pretrained=True)
        
        # Remove the classification layer (avgpool and fc) at the end
        modules = list(resnet.children())[:-2]
        self.resnet = nn.Sequential(*modules)
        self.fc = nn.Linear(resnet.fc.in_features, embedding_size)
        
        # Freeze or unfreeze ResNet layers based on train_CNN flag
        if not train_CNN:
            for param in self.resnet.parameters():
                param.requires_grad = False

    def forward(self, images):
        features = self.resnet(images)
        features = features.reshape(features.shape[0] , features.shape[1] , -1)
        features = features.permute(0 , 2 , 1) # (B , H*W , 2048)
        features = self.fc(features) # (B , H*W , embeding_size)
        return features    
class PrunedEncoderCNN(nn.Module):
    def __init__(self, model_path):
        super(PrunedEncoderCNN, self).__init__()
        self.model = torch.load(model_path)
    
    def forward(self, images):
       features = self.model(images)
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
class PositionalEncoding_2D(nn.Module):
    def __init__(self, d_model: int, height: int = 8, width: int = 8, dropout_prob: float = 0.1):
        super(PositionalEncoding_2D, self).__init__()
        self.d_model = d_model
        self.height = height
        self.width = width
        self.dropout = nn.Dropout(dropout_prob)

        pe = torch.zeros(d_model, height, width)
        d_model = d_model // 2
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        position_w = torch.arange(0, width).unsqueeze(1)
        position_h = torch.arange(0, height).unsqueeze(1)

        pe[0:d_model:2, :, :] = torch.sin(position_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
        pe[1:d_model:2, :, :] = torch.cos(position_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
        pe[d_model::2, :, :] = torch.sin(position_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)
        pe[d_model + 1::2, :, :] = torch.cos(position_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)

        pe = pe.view(height, width, self.d_model)
        pe = pe.unsqueeze(0)

        self.register_buffer('pe', pe)

    def forward(self, x):
        if self.pe.device != x.device:
            self.pe = self.pe.to(x.device)
        x = x.permute(0 , 2 , 3 , 1)
        x = x + self.pe[:, :x.shape[1], :x.shape[2], :].requires_grad_(False)
        x = x.permute(0 , 3 , 1 , 2)
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
        encoder_output = x.permute(1 , 0 , 2) # (seq_len x batch_size x embeding_dim)
        x = self.positional_encoding(x) # (1 x batch_size x embedding_dim)
        encoder_output = self.encoder(encoder_output)
        return encoder_output
class TransformersDecoder(nn.Module):
    def __init__(self,embeding_size,trg_vocab_size,num_heads,num_decoder_layers,dropout):
        super(TransformersDecoder,self).__init__()
        
        self.num_heads = num_heads
        self.embedding = nn.Embedding(trg_vocab_size,embeding_size)
        self.pos = PositionalEncoding(d_model = embeding_size)
        self.decoder_layer = nn.TransformerDecoderLayer(d_model=embeding_size, nhead=num_heads)
        self.decoder= nn.TransformerDecoder(self.decoder_layer, num_layers=num_decoder_layers)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.linear = nn.Linear(embeding_size , trg_vocab_size)
        self.drop = nn.Dropout(dropout)
        
        
    def make_mask(self,sz):
        mask = torch.zeros((sz,sz), dtype=torch.float32)
        for i in range(sz):
            for j in range(sz):
                if j > i: mask[i][j] = float('-inf')
        return mask
    
    def forward(self,features,caption):
        
        tgt_seq_length , N =caption.shape
        embed = self.drop(self.embedding(caption))
        embed = self.pos(embed)
        trg_mask = self.make_mask(tgt_seq_length).to(self.device)
        decoder = self.decoder(tgt = embed , memory = features , tgt_mask = trg_mask )
        output = self.linear(decoder)
        return output
               
class EncodertoDecoder(nn.Module):
    def __init__(self,embeding_size=512,trg_vocab_size=2992,num_heads=8,num_decoder_layers=4,dropout=0.2):
        super(EncodertoDecoder,self).__init__()

        self.image_encoder = EncoderCNN(embeding_size)
        self.encoder = TransformersEncoder(embeding_size, num_heads, 4 , dropout)
        self.decoder = TransformersDecoder(embeding_size, trg_vocab_size, num_heads, num_decoder_layers, dropout)
                                                
    def forward(self , image , caption):
                
        features = self.image_encoder(image) #This one is for without pruning
        #features = self.pruned_image_encoder(image) # This one is for pruned_model
        features = self.encoder(features)
        output = self.decoder(features , caption)
        return output
       