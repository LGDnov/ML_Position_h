#import libraries
import torch
import torchvision.models as models
import torch.nn as nn


def model_defin():
    # Model definition:
    # Get pretrained model using torchvision.models as models library
    model = models.densenet161(pretrained=True)

    # Turn off training for their parameters
    for param in model.parameters():
       param.requires_grad = False

    # Create new classifier for model using torch.nn as nn library
    classifier_input = model.classifier.in_features
    num_labels = 4
    classifier = nn.Sequential(nn.Linear(classifier_input, 256),
                           nn.ReLU(),
                           nn.Linear(256,128),
                           nn.ReLU(),
                           nn.Linear(128, num_labels),
                           nn.LogSoftmax(dim=1))
    # Replace default classifier with new classifier
    model.classifier = classifier
    # Find the device available to use using torch library
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Move model to the device specified above
    model.to(device)
    return model



