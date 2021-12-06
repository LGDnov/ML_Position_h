#import libraries
import matplotlib.pyplot as plt
import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.optim as optim
import trtorch
from PIL import Image
import matplotlib.pyplot as plt
import create_model


#determination of the mean and standard deviation before normalization
def find_mean_std(img_path):
    transform = transforms.Compose([
    transforms.ToTensor()
    ])
    img = Image.open(img_path)
    # convert PIL image to numpy array
    img_tr = transform(img)
    mean, std = img_tr.mean([1,2]), img_tr.std([1,2])
    return mean, std

m_s = find_mean_std("Source/Data_set/Train/front/0.jpg")
transformations = transforms.Compose([
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=m_s[0], std=m_s[1])
])
# loading data
train_set = datasets.ImageFolder("Source/Data_set/Train", transform = transformations)
val_set = datasets.ImageFolder("Source/data-set-01/Data_set/Valid", transform = transformations)
train_loader = torch.utils.data.DataLoader(train_set, batch_size=100, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_set, batch_size =100, shuffle=True)


#Model definition
model = create_model.model_defin()
# Find the device available to use using torch library
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Set the error function using torch.nn as nn library
criterion = nn.NLLLoss()
# Set the optimizer function using torch.optim as optim library
optimizer = optim.Adam(model.classifier.parameters())
epochs = 4
epochs_1 =[]
train_losses = []
valid_losses = []
acc_list = []

for epoch in range(epochs):
    train_loss = 0
    val_loss = 0
    accuracy = 0
    
    # Training the model
    model.train()
    counter = 0
    for inputs, labels in train_loader:
        # Move to device
        inputs, labels = inputs.to(device), labels.to(device)
        # Clear optimizers
        optimizer.zero_grad()
        # Forward pass
        output = model.forward(inputs)
        # Loss
        loss = criterion(output, labels)
        # Calculate gradients (backpropogation)
        loss.backward()
        # Adjust parameters based on gradients
        optimizer.step()
        # Add the loss to the training set's rnning loss
        train_loss += loss.item()*inputs.size(0)
        
        # Print the progress of our training
        counter += 1
        if counter%80==0:
            print(counter, "/", len(train_loader))
        
    # Evaluating the model
    model.eval()
    counter = 0
    # Tell torch not to calculate gradients
    with torch.no_grad():
        for inputs, labels in val_loader:
            # Move to device
            inputs, labels = inputs.to(device), labels.to(device)
            # Forward pass
            output = model.forward(inputs)
            # Calculate Loss
            valloss = criterion(output, labels)
            # Add loss to the validation set's running loss
            val_loss += valloss.item()*inputs.size(0)
            
            # Since our model outputs a LogSoftmax, find the real 
            # percentages by reversing the log function
            output = torch.exp(output)
            # Get the top class of the output
            top_p, top_class = output.topk(1, dim=1)
            # See how many of the classes were correct?
            equals = top_class == labels.view(*top_class.shape)
            # Calculate the mean (get the accuracy for this batch)
            # and add it to the running accuracy for this epoch
            accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
            
            # Print the progress of our evaluation
            counter += 1
            if counter%50==0:
                 print(counter, "/", len(val_loader))
                 print('accuracy = ',torch.mean(equals.type(torch.FloatTensor)).item())
           
    
    # Get the average loss for the entire epoch
    train_loss = train_loss/len(train_loader.dataset)
    valid_loss = val_loss/len(val_loader.dataset)
    acc_list.append(accuracy/len(val_loader))
    # Print out the information
    print('Accuracy: ', accuracy/len(val_loader))
    print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(epoch, train_loss, valid_loss))
    epochs_1.append(epoch)
    train_losses.append(train_loss)
    valid_losses.append(valid_loss)
  

FILE = "model/model_my01.pth"
torch.save(model,FILE)

plt.plot(epochs_1,train_losses,label = "Train")
plt.plot(epochs_1,valid_losses,label = "Valid")
plt.plot(epochs_1,acc_list,label = "Acc")
plt.legend()
plt.show()