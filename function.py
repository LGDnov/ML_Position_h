#import libraries
import time
import torch
import torchvision.transforms as transforms
from torch2trt import torch2trt
from PIL import Image
from os import listdir

def image_show(image):
   # a = plt.imread(image)
   # plt.imshow(a)
   # plt.show()                                                                          
   img = Image.open(image)
   img.show() 

transformations = transforms.Compose([
      transforms.Resize((256,256)),
      transforms.ToTensor(),
      transforms.Normalize(mean=[0.8443, 0.7971, 0.7905], std=[0.2109, 0.2767, 0.2842])
      ])

#loading model
def loading(PATH, value=1):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.load(PATH, map_location = device)
    if value == 1:
       model.eval()
    return model

#prediction image
def predict(image_file, model, show=0):
    if show==1:
      image_show(image_file)
    image  = Image.open(image_file)
    # Give image to model to predict output
    picture = transformations(image)
    input = picture.unsqueeze(0)
    # Pass the image through our model
    output = model.forward(input)
    # Reverse the log function in our output
    output = torch.exp(output)
    # Get the top predicted class, and the output percentage for
    # that class
    probs, classes = output.topk(1, dim=1)
    top_prob = probs.item()
    top_class = classes.item()
    # Print the results
    if top_class == 0:
       top_class='back'
    if top_class == 1:
       top_class='front'
    if top_class == 2:
       top_class='full'
    if top_class == 3:
       top_class='side'
    print("The model is ", top_prob*100, "% certain that the image: ",image_file, "has a predicted class of ", top_class) 
    return top_class

#prediction for a set of images in a folder
def predict_m(folder, type_class, model):
   list_name = listdir(folder)
   k=0
   list_name_folse= []
   t0 = time.time()
   for i  in list_name:
      tc = predict(folder+"/"+i, model,show = 0)
      if tc==type_class:
         k = k+1
      else:
         list_name_folse.append(i)
   t1 = time.time()
   fps = len(list_name) / (t1 - t0)
   print("Speed, fps = ",fps," Time = ",(t1 - t0))
   print("Number of faithful: ",k/len(list_name)*100,"%"," total: ",len(list_name))
   print("Unfaithful:")
   for i in list_name_folse:
      print(i)


#converter Tensor_RT
def Convert_TRT(model,Example_img):
    image  = Image.open(Example_img)
    picture = transformations(image)
    tensor = picture.unsqueeze(0) 
    model_trt = torch2trt(model, tensor)
    return model_trt
    


