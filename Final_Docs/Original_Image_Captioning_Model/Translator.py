# -*- coding: utf-8 -*-
"""
Created on Fri Jul  5 05:33:18 2024

@author: jishu
"""
import torch
import torch.nn as nn
import torchvision.transforms as transforms

import random
import os

from translate import Translator
from get_loader import get_loader
from PIL import Image


def caption_generate(model,dataset,image,device,max_length = 50):
    outputs=[dataset.vocab.stoi["<SOS>"]]
    for i in range(max_length):
        trg_tensor =torch.LongTensor(outputs).unsqueeze(1).to(device)
        image = image.to(device)
        
        with torch.no_grad():
            output = model(image,trg_tensor)
            
        best_guess = output.argmax(2)[-1, :].item()
        outputs.append(best_guess)
        
        if best_guess == dataset.vocab.stoi["<EOS>"]:
            break
    caption = [dataset.vocab.itos[idx] for idx in outputs]
    
    return caption[1:]

def translate_to_korean(sentence):
    if isinstance(sentence, list):
        sentence = ' '.join(sentence)
    translator = Translator(to_lang="ko")
    translation = translator.translate(sentence)
    return translation

if __name__ == "__main__":
    
    #Load the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load the model
    model_path = r'unpruned_model.pth'
    model = torch.load(model_path)
    model = model.to(device)
    model.eval()
    
    # Transform 
    transform = transforms.Compose([transforms.Resize((350,350)),
                                transforms.RandomCrop((256,256)),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    
    # Hyperparameters
    images_path , caption_path = r"C:\Users\DELL\Desktop\Jishu\flickr8k\Images" , r"C:\Users\DELL\Desktop\Jishu\flickr8k\captions.txt"
    
    BATCH_SIZE = 32
    validation_dataloader , validation_dataset = get_loader(images_path,caption_path ,transform,batch_size = BATCH_SIZE,num_workers=4 , train = False)
    
    
    # Now make the predictions
    
    num_samples = len(validation_dataset)
    
    random_indices = random.sample(range(num_samples), 10)
    
    # Now make the predictions
    
    num_samples = len(validation_dataset)
    
    random_indices = random.sample(range(num_samples), 10)
    
    
    for idx in random_indices:
        _ , caption = validation_dataset[idx]
        image_id = validation_dataset.imgs[idx]
        image=transform(Image.open(os.path.join(images_path,image_id)).convert("RGB")).unsqueeze(0)
        image_name = os.path.basename(image_id).split('.')[0]
        
        # Now we have to send the image to the generate caption function and the get a predicted output
        correct_label = ""
        correct_caption = caption.detach().cpu().numpy()
        
        for i in correct_caption:
            token = validation_dataset.vocab.itos[i]
            correct_label += token
            correct_label += " " 
            
            if i == 2:
                break 
            
        english_sentence = caption_generate(model, validation_dataset, image, device)
        korean_sentence = translate_to_korean(english_sentence)
        
        print(
            "Example OUTPUT ENG: "
            + " ".join(english_sentence)
        )
        print("\n")
        print(
            "Example OUTPUT KOR: "
            + " ".join(korean_sentence)
        )
        
    


    

