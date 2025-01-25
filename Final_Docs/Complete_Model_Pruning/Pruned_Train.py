# -*- coding: utf-8 -*-
"""
Created on Wed Jun 12 11:35:35 2024

@author: jishu
"""

import torch
import torch.optim as optim
from get_loader import get_loader
from torchvision import transforms
import New_Pruned_Model
import torch.nn as nn
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from utils import print_examples
from New_Pruned_Model import EncoderCNN




mse_loss = nn.MSELoss()


def match_hidden_states(sub_network_hidden_states, decoder_hidden_states , num_selected_layers = 3):
    # Here we have to select the layers coming with decoder network
    num_total_layers = len(decoder_hidden_states)
    selected_layers_indices = [num_selected_layers - 1]
    for i in range(num_selected_layers - 1):
        selected_layers_indices.append(int((num_total_layers - 1) * (i + 1)/num_selected_layers))
    
    selected_decoder_hidden_states = [decoder_hidden_states[idx] for idx in selected_layers_indices]
    loss = 0 
    for sub_state, dec_state in zip(sub_network_hidden_states, selected_decoder_hidden_states):
        loss += mse_loss(sub_state, dec_state)
    return loss


def train():

    transform = transforms.Compose([transforms.Resize((350,350)),
                                transforms.RandomCrop((256,256)),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    images_path , caption_path = r"C:\Users\DELL\Desktop\Jishu\flickr8k\Images" , r"C:\Users\DELL\Desktop\Jishu\flickr8k\captions.txt"
    pruned_resnet_model_path = r"Pruned_ResNet\fine_tuned_model.pth"
    
    BATCH_SIZE = 32
    data_loader , dataset = get_loader(images_path,caption_path ,transform,batch_size = BATCH_SIZE,num_workers=4)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_epochs = 20
    learning_rate = 3e-4
    trg_vocab_size = len(dataset.vocab)
    embedding_size = 512
    num_heads = 8
    num_decoder_layers = 2
    dropout = 0.10
    pad_idx=dataset.vocab.stoi["<PAD>"]
    save_model = True
    writer =SummaryWriter("runs/loss_plot")
    step = 0
    encoder_regularization_penalty = 0.01
    decoder_regularizartion_penalty = 0.01 
    
    model = torch.load(r'Models\Only_ResNet_Pruned.pth', map_location=device)
    model = model.to(device)
    
    # Now we define the pruned model
    pruned_model = New_Pruned_Model.EncodertoDecoder(embeding_size=embedding_size,
                            trg_vocab_size=trg_vocab_size,
                            num_heads=num_heads,
                            num_decoder_layers=num_decoder_layers,
                            dropout=dropout , pruned_resnet_model_path = pruned_resnet_model_path).to(device)
    
    
    optimizer = optim.Adam(pruned_model.parameters(),lr = learning_rate)
    criterion2 = nn.CrossEntropyLoss(ignore_index=pad_idx)
    
    l = []
    
    for epoch in range(num_epochs):
        
        print(f"[Epoch {epoch} / {num_epochs}]")
        
        
        
        model.eval()
        pruned_model.train()
        Total_loss = 0.0
        for idx, (images, captions) in tqdm(enumerate(data_loader), total=len(data_loader), leave=False):
            
            images = images.to(device)
            captions = captions.to(device)
            
            with torch.no_grad():
                output , hidden_original_decoder_outputs = model(images, captions[:-1])
            pruned_model_outputs , hidden_pruned_decoder_outputs = pruned_model(images , captions[:-1])
            
            
            #print(pruned_model_outputs.shape)
            
            #outputs = output.reshape(-1, output.shape[2])
            pruned_model_outputs = pruned_model_outputs.reshape(-1 , pruned_model_outputs.shape[2])
            
            
            
            target = captions[1:].reshape(-1).to(device)
         
            # Now calculate two losses : One associated with the general targets and another with the hidden_state_losses of the decoder
            optimizer.zero_grad()
            
            loss_match = criterion2(pruned_model_outputs , target) # This will be the similarity between the outputs of the original model and the pruned model
            
            # Compute the L2 Regularization loss of the encoder weights
            l2_reg = pruned_model.encoder.compute_penalty(encoder_regularization_penalty)
            
            mse_loss = match_hidden_states(hidden_pruned_decoder_outputs,hidden_original_decoder_outputs)
            dec_loss = mse_loss * decoder_regularizartion_penalty
            
            # Now calculate the total loss
            total_loss = loss_match + l2_reg + dec_loss
            
            lossofepoch = total_loss.item()
            Total_loss += lossofepoch
            
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(pruned_model.parameters(),max_norm=1)
            
            optimizer.step()
            writer.add_scalar("Training Loss",Total_loss,global_step=step)
            step+=1
            
        LOSS = Total_loss / len(data_loader)
        l.append(LOSS) 
        print("Loss of the epoch is", Total_loss / len(data_loader))
        if save_model and epoch == 19:
            torch.save(pruned_model , 'model_final_4_2_T.pth')
            

        pruned_model.eval()
        print_examples(pruned_model, device, dataset)
            
        



if __name__ == "__main__":
    train()
            
        
        