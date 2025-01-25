import os
import torch
import torchvision.models as models
import torch.nn as nn
import torch_pruning as tp
import Tiny_ImageNet_Loader
from torchvision import transforms
from Cnn_Pruning.get_loader import get_loader
from tqdm import tqdm


# First define the original model. The pruned model and the original model will be instances of this
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
    
class MySlimmingPruner(tp.pruner.MetaPruner):
    def regularize(self , model , reg):
        for m in model.modules():
            if isinstance(m , (nn.BatchNorm1d , nn.BatchNorm2d , nn.BatchNorm3d)) and m.affine == True:
                m.weight.grad.data.add_(reg*torch.sign(m.weight.data))                
class MySlimmingImportance(tp.importance.Importance):
    def __call__(self , group):
        group_imp = []
        for dep , idx in group:
            layer = dep.target.module
            prune_fn = dep.handler
            if isinstance(layer, (nn.BatchNorm1d , nn.BatchNorm2d , nn.BatchNorm3d)) and layer.affine:
                local_imp = torch.abs(layer.weight.data)
                group_imp.append(local_imp)
        if(len(group_imp) == 0): return None
        
        group_imp = torch.stack(group_imp , dim = 0).mean(dim = 0)
        return group_imp # This has the dimension(num_channels , _)
    
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    exit()

    original_model = EncoderCNN().to(device)
    pruned_model = EncoderCNN(train_CNN=True).to(device)  # Train_CNN is set to true because we want to change its parameters
    
    # Set the importance Criteria
    imp = MySlimmingImportance()

    # Define the data transforms
    transform = transforms.Compose([
        transforms.Resize(350),
        transforms.CenterCrop(256),
        transforms.ToTensor()])

    # Define the paths for images and captions 
    images_path = r"D:\ML\Korea\Jishu\Jishu\rsicd\images"
    caption_path = r"D:\ML\Korea\Jishu\Jishu\rsicd\captions.csv"
    
    # Also load the TinyImageNet dataset
    tiny_imagenet_path = r"D:\ML\Korea\Jishu\Jishu\Pruning\tiny-imagenet-200"
    train_loader , train_dataset = Tiny_ImageNet_Loader.get_loader(root_folder = tiny_imagenet_path, transform = transform)
    
    # Get the data loader
    fine_tune_loader, fine_tune_dataset = get_loader(images_path, caption_path, transform)

    # Define the layers which need to be ignored
    ignored_layers = []
    for m in original_model.modules():
        if isinstance(m, torch.nn.Linear) and m.out_features == 512:
            ignored_layers.append(m)

    # Define example input
    example_inputs = torch.randn(1, 3, 256, 256).to(device)

    # Pruner initialization
    iterative_steps = 5  # Number of iterations to achieve target pruning ratio
    pruner = MySlimmingPruner(
        pruned_model, 
        example_inputs, 
        global_pruning=False,  # If False, a uniform ratio will be assigned to different layers.
        importance=imp,  # Importance criterion for parameter selection
        iterative_steps=iterative_steps,  # The number of iterations to achieve target ratio
        pruning_ratio=0.5,  # Remove 50% channels
        ignored_layers=ignored_layers
    )

    # Create the directory for saving models if it doesn't exist
    save_dir = "Pruned_Resnet"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Now we perform the training phase
    # Set hyperparameters
    num_epochs = 7
    fine_tune_epochs = 1
    learning_rate = 1e-5
    criterion = nn.MSELoss()
    regularizer = 1e-5
    optimizer = torch.optim.Adam(pruned_model.parameters(), lr=learning_rate)

    # Training loop
    for epoch in range(num_epochs):
        pruned_model.train()
        train_loader_iter = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs}', leave=False)
        for idx, (images, _) in enumerate(train_loader_iter):
            images = images.to(device)

            true_outputs = original_model(images)
            pred_outputs = pruned_model(images)

            loss = criterion(pred_outputs, true_outputs)

            optimizer.zero_grad()
            loss.backward()
            pruner.regularize(pruned_model, reg=regularizer)
            optimizer.step()
            train_loader_iter.set_postfix({'Loss': loss.item()})

    # Save the pruned model
    torch.save(pruned_model, os.path.join(save_dir, 'pruned_model.pth'))
    print("Pruned model saved.")

    # Pruning and fine-tuning iteration
    base_macs, base_nparams = tp.utils.count_ops_and_params(pruned_model, example_inputs)
    for i in range(iterative_steps):
        pruner.step()

        macs, nparams = tp.utils.count_ops_and_params(pruned_model, example_inputs)
        print(pruned_model(example_inputs).shape)
        print(
            "  Iter %d/%d, Params: %.2f M => %.2f M"
            % (i+1, iterative_steps, base_nparams / 1e6, nparams / 1e6)
        )
        print(
            "  Iter %d/%d, MACs: %.2f G => %.2f G"
            % (i+1, iterative_steps, base_macs / 1e9, macs / 1e9)
        )
        print("="*16)
        # Fine-tune your model here
        for epoch_ft in range(fine_tune_epochs):
            fine_tune_loader_iter = tqdm(fine_tune_loader, desc=f'Fine-tuning Epoch {epoch_ft + 1}/{fine_tune_epochs}', leave=False)
            for images, _ in fine_tune_loader_iter:
                images = images.to(device)

                true_outputs = original_model(images)
                pred_outputs = pruned_model(images)

                loss = criterion(pred_outputs, true_outputs)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                fine_tune_loader_iter.set_postfix({'Loss': loss.item()})
    
    # Save the fine-tuned model
    torch.save(pruned_model, os.path.join(save_dir, 'fine_tuned_model.pth'))
    print("Fine-tuned model saved.")
