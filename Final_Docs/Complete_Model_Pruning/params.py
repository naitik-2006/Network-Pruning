import torch 
import os
import New_Pruned_Model

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() )

un_pruned_model = torch.load(r'D:\ML\Korea\Jishu\Jishu\Final_Docs\Original_Image_Captioning_Model\model.pth', map_location=device)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
un_pruned_model = un_pruned_model.to(device)

print("Un Pruned Model :-", count_parameters(un_pruned_model))

pruned_model = torch.load(r'D:\ML\Korea\Jishu\Jishu\Final_Docs\Complete_Model_Pruning\model_final_4_2_T.pth', map_location=torch.device('cpu'))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Pruned Model :-", count_parameters(pruned_model))