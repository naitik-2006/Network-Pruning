import torch
import torch.optim as optim
from get_loader import get_loader
from torchvision import transforms
from Model import EncodertoDecoder
import torch.nn as nn
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from utils import  print_examples
from Model import EncoderCNN

def train():

    transform = transforms.Compose([transforms.Resize((350,350)),
                                transforms.RandomCrop((256,256)),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    images_path , caption_path = r"D:\ML\Korea\Jishu\Jishu\rsicd\images" , r"D:\ML\Korea\Jishu\Jishu\rsicd\captions.csv"
    
    BATCH_SIZE = 32
    data_loader , dataset = get_loader(images_path,caption_path ,transform,batch_size = BATCH_SIZE,num_workers=4)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_epochs = 15
    learning_rate = 3e-4
    trg_vocab_size = len(dataset.vocab)

    embedding_size = 512
    num_heads = 8
    num_decoder_layers = 4
    dropout = 0.20
    pad_idx=dataset.vocab.stoi["<PAD>"]
    save_model = True
    writer =SummaryWriter("runs/loss_plot")
    step = 0

    model = EncodertoDecoder(embeding_size=embedding_size,
                            trg_vocab_size=trg_vocab_size,
                            num_heads=num_heads,
                            num_decoder_layers=num_decoder_layers,
                            dropout=dropout).to(device)
    optimizer = optim.Adam(model.parameters(),lr = learning_rate)

    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)


    for epoch in range(num_epochs):
        
        print(f"[Epoch {epoch} / {num_epochs}]")
        
        model.eval()
        print_examples(model, device, dataset)
        
        model.train()
        total_loss = 0.0
        for idx, (images, captions) in tqdm(enumerate(data_loader), total=len(data_loader), leave=False):
            images = images.to(device)
            captions = captions.to(device)
            
            output = model(images, captions[:-1])
            output = output.reshape(-1, output.shape[2])
            target = captions[1:].reshape(-1)
            optimizer.zero_grad()
            loss = criterion(output,target)
            lossofepoch = loss.item()
            total_loss += lossofepoch
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),max_norm=1)
            
            optimizer.step()
            writer.add_scalar("Training Loss",loss,global_step=step)
            step+=1
            
        print("Loss of the epoch is", total_loss / len(data_loader))
        if save_model and (epoch%14 == 0):
            torch.save(model , 'model.pth')
            
        model.eval()
        print_examples(model, device, dataset)



if __name__ == "__main__":
    train()
            
        
        