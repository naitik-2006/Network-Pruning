import pandas as pd
from PIL import Image
from torch.utils.data import DataLoader ,Dataset
import spacy 
import torch
import os
from torch.nn.utils.rnn import pad_sequence
from sklearn.model_selection import train_test_split

spacy_eng=spacy.load("en_core_web_sm")

# Now we have to change to handle train and test cases
class Vocabulary:
    def __init__(self,freq_threshold):
        self.itos={0:"<PAD>" , 1 :"<SOS>" , 2: "<EOS>" , 3:"<UNK>"}
        self.stoi={"<PAD>":0 ,"<SOS>": 1 ,"<EOS>":2,"<UNK>":3}
        self.freq_threshold=freq_threshold
        
    def __len__(self):
        return len(self.itos)
    
    def tokenizer_eng(self,text):
        return [tok.text.lower() for tok in spacy_eng.tokenizer(text)]
    
    def build_vocabulary(self,sentence_list):
        frequencies={}
        idx=4
        
        for sentence in sentence_list:
            for word in self.tokenizer_eng(sentence):
                if word not in frequencies:
                    frequencies[word]=1
                else:
                    frequencies[word]+=1
                    
                if frequencies[word] == self.freq_threshold:
                    self.stoi[word]=idx
                    self.itos[idx]=word
                    idx+=1
                    
    def numericalize(self,text):
        tokenized_text = self.tokenizer_eng(text)
        result = []

        for token in tokenized_text:
            if token in self.stoi:
                result.append(self.stoi[token])
            else:
                result.append(self.stoi["<UNK>"])
        
        return result

class FlickerData(Dataset):
    def __init__(self,root_dir,caption_file,transform=None,freq_threshold=5 , train = True):
        self.root_dir=root_dir
        self.df=pd.read_csv(caption_file)
        self.transform=transform
        self.train = train
        
        # Split into train and test images
        self.train_df , self.test_df = train_test_split(self.df, test_size = 0.1 , random_state = 42)
        
        if(self.train == True):
            self.imgs = self.train_df["image"].reset_index(drop = True)
            self.captions = self.train_df["caption"].reset_index(drop = True)
        else:
            self.imgs = self.test_df["image"].reset_index(drop = True)
            self.captions = self.test_df["caption"].reset_index(drop = True)
                    
        self.caption=self.df["caption"]
        self.vocab=Vocabulary(freq_threshold)
        self.vocab.build_vocabulary(self.caption.tolist())
        
    def __len__(self):
        if(self.train):
            return len(self.train_df)
        else:
            return len(self.test_df)
    
    def __getitem__(self,index):
        caption=self.captions[index]
        image_id=self.imgs[index]
        img=Image.open(os.path.join(self.root_dir,image_id)).convert("RGB")
        
        if self.transform is not None:
            img=self.transform(img)
            
        numericalized_caption=[self.vocab.stoi["<SOS>"]]
        numericalized_caption+=self.vocab.numericalize(caption)
        numericalized_caption.append(self.vocab.stoi["<EOS>"])
        
        return img,torch.tensor(numericalized_caption)

class MyCollate:
    def __init__(self,pad_idx):
        self.pad_idx = pad_idx
        
    def __call__(self,batch):
        imgs=[item[0].unsqueeze(0) for item in batch]
        imgs=torch.cat(imgs,dim=0)
        target = [item[1] for item in batch]
        target = pad_sequence(target,batch_first=False , padding_value=self.pad_idx)
        
        return imgs,target
    
def get_loader(
    root_folder,
    caption_file,
    transform,
    batch_size=32,
    num_workers=8,
    shuffle=True,
    pin_memory=True,
    train = True
):

    dataset = FlickerData(root_folder,caption_file,transform=transform , train = train)
    pad_idx=dataset.vocab.stoi["<PAD>"]
    
    if(train == False):
        batch_size = 1
    
    loader = DataLoader(
        dataset = dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        pin_memory=pin_memory,
        collate_fn=MyCollate(pad_idx=pad_idx),
    )
    return loader,dataset
