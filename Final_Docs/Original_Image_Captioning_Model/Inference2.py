# -*- coding: utf-8 -*-
"""
Created on Mon May 27 16:57:52 2024

@author: jishu
"""
import torch
import json

from rouge_score import rouge_scorer
from pycocoevalcap.cider import cider

from get_loader import get_loader
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter

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

def run_validation(model, validation_dataloader, validation_dataset, max_len, device, writer):
    model.eval()
    count = 0

    expected = []
    predicted = []
    results_dict = {}

    with torch.no_grad():
        for idx , (image , caption) in enumerate(validation_dataloader):
            
            count += 1
            image = image.to(device)
            #encoder_mask = batch["decoder_mask"].to(device) # (b, 1, 1, seq_len)
            
            # Check that the batch size is 1
            assert image.size(0) == 1, "Batch size must be 1 for validation"

            print("Processing Image:", count)
            model_out = caption_generate(model, validation_dataset , image , device , max_len)

            # Convert PyTorch tensors to NumPy arrays
            target_text = caption.detach().cpu().numpy().tolist()
            target_text_flat = [token for sublist in target_text for token in sublist]
            
            # Initialize strings to store the predicted and target text
            model_out_text = ""
            target_text_2 = ""

            # Iterate over the predicted tokens
            for i in model_out:
                 
                token = i
                if token == '<EOS>':
                    break
                model_out_text += token + " "

            # Iterate over the target tokens
            for i in target_text_flat:
                token = validation_dataset.vocab.itos[i]
                if token == '<EOS>':
                    break
                target_text_2 += token + " "


            expected.append(target_text_2.strip())
            predicted.append(model_out_text.strip())
            results_dict = {}  # Initialize an empty dictionary

            results_dict[target_text_2.strip()] = model_out_text.strip()

            # Alternatively, if you have a loop for multiple pairs, you can use:
            
            print("Expected :- ", target_text_2)
            print("Predicted :- ", model_out_text)
            results_dict[idx] = [expected,predicted]
        
    if writer:
        # Compute the ROUGE scores
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        rouge1, rouge2, rougeL = 0, 0, 0
        
        cider_scorer = cider.Cider()
        ciders = 0
        
        # Dictionaries to store the ground truths (gts) and predictions (res) for CIDEr scoring
        gts = {}
        res = {}
        
        for idx, (ref, pred) in enumerate(zip(expected, predicted)):
            # Ensure ref is a list of sentences
            if isinstance(ref, str):
                ref = [ref]
    
            # Ensure pred is a list containing a single sentence
            if isinstance(pred, str):
                pred = [pred]

            # Sanity check
            assert(type(ref) is list)
            assert(len(ref) > 0)
            assert(type(pred) is list)
            assert(len(pred) == 1)

            # Fill the gts and res dictionaries for CIDEr scoring
            gts[idx] = ref
            res[idx] = pred

        for ref, pred in zip(expected, predicted):
            
            scores = scorer.score(ref, pred)
            rouge1 += scores['rouge1'].fmeasure
            rouge2 += scores['rouge2'].fmeasure
            rougeL += scores['rougeL'].fmeasure
            
       
        # Compute CIDEr score for the entire corpus
        cider_score, cider_scores = cider_scorer.compute_score(gts, res)
        ciders += cider_score

        rouge1 /= len(expected)
        rouge2 /= len(expected)
        rougeL /= len(expected)

        writer.add_scalar('validation ROUGE1', rouge1)
        writer.add_scalar('validation ROUGE2', rouge2)
        writer.add_scalar('validation ROUGEL', rougeL)
        writer.flush()
        
        
        
        print("ROUGE-1 = ", rouge1)
        print("ROUGE-2 = ", rouge2)
        print("ROUGE-L = ", rougeL)
        print("CIDEr = ", ciders)
        
    with open("results.json", "w") as json_file:
        json.dump(results_dict, json_file, indent=4)

# The rest of your code, such as data loading and model definition, goes here.

if __name__ == "__main__":
    # Define the hyperparameters
    transform = transforms.Compose([transforms.Resize((350,350)),
                                transforms.RandomCrop((256,256)),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    images_path , caption_path = r"D:\ML\Korea\Jishu\Jishu\rsicd\images" , r"D:\ML\Korea\Jishu\Jishu\rsicd\captions.csv"
    
    BATCH_SIZE = 32
    validation_dataloader , validation_dataset = get_loader(images_path,caption_path ,transform,batch_size = BATCH_SIZE,num_workers=4 , train = False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_epochs = 15
    learning_rate = 3e-4
    trg_vocab_size = len(validation_dataset.vocab)

    embedding_size = 512
    num_heads = 8
    num_decoder_layers = 4
    dropout = 0.20
    pad_index=validation_dataset.vocab.stoi["<PAD>"]
    save_model = True
    writer =SummaryWriter("runs/loss_plot")
    step = 0
    max_len = 50
    
    
    # Now we load the model
    model = torch.load('model.pth')
    model = model.to(device)
    
    # Initialize the tensorboard
    logs_dir = "logs"
    writer = SummaryWriter(logs_dir)

    # Now we have to send these dataset and dataloaders to the run_validation function
    
    run_validation(model, validation_dataloader, validation_dataset, max_len, device , writer)
    
    
    
    
    
    
    
