# FAISS FAST API
## Facebook Marketplace Image-based Recommendation

In this project, we have implemented the FAST API in a docker container, to be deployed in the cloud. The API provides a method that is categorizing images into 13 product categories and searches for similar images from the image database. The model is based on ResNet50 neural networks and indexing is performed by the FAISS indexing.

Key technologies used: Resnet50 neural network (Pytorch), FAISS indexing, FastAPI, Docker 

## Model 

The model used is the ResNet50 neural network, which is a CNN and can be used for image classification. to make efficient categorization we used the transfer learning approach and load the weights of the resnet50 model "IMAGENET1k_V2" which is trained to perform classification on 1K different classes. As we only had 13 classes, we need to resize the model classification layers and unfreeze the last 2 layers of the pre-trained model. This way our model adopts more to our database.

## Data preparation

The raw data contains two ".csv" tables and one "*.jpg" image archive. The first table "Products.csv" lists the market products grouped by listing ID ("product_id") with their classification and description, and the second table "Images.csv" maps the listing ID to the image_id corresponding to the label of the saved image.

Let's start with processing text data. The cleanup starts by converting the "price" column to the correct "float" and removing all raw data, consisting of missing data or NaN data, from the "Products.csv" table. The processing is performed in the script "clean_tabular_data.py". The "category" field contains a hierarchical structure separated by " / ". To train the model, we need to extract the root category and give each unique category an integer. The "Image.csv" dataset maps products to images, where there are two images for each product. These images are photographs of products from different angles, so they may look very similar. Finally, we join the two tables on the "product_id" key, forming a dataset that displays the image tag with its category. The transformations described can be found in "sandbox.ipynb". Our images dataset contains "*.jpg" files of different resolution and aspect ratios. As the resnet50 is trained by the images of size 224x224, we need to transform it into the right resolution. The processing is performed in the script "clean_images.py"


## Model Training
The initial dataset of 12604 categorized images is split into 'evaluation' (10%), 'training' (80%), and test (10%) for each batch from the data loader which has a batch size of 200.

During the training procedure model update weights coefficients based on its performance on the training dataset. Each epoch (the round of iterations across 'training' data) we test model performance on the 'evaluation' dataset. As soon as we proceed through the desired number of epochs we test our model on our 'test' dataset, which is our final performance indicator. 

### Dataloader
The dataloader used in the standard training routine (torch.utils.data.DataLoader) is the wrapper around torch.utils.data.Dataset object. Therefore we create a class inheriting torch.utils.data.Dataset, where we implement datahandling with respect to the dataloader spec. We use a data augmentation procedure to increase model resistance to overfitting, so each time image passed to training it experiences random rotation and horizontal/vertical flips. Such transformations applied to the 'training' dataset effectively increase its size, but during 'test' and 'evaluation' procedures it provides extra noise making it hard to analyse results.

### Training procedure

The model training requires a measure of the model performance, here we use the so-called cross-entropy losses criterion, which is standard for image classification procedures. Then to provide feedback on the model we use the ADAM method, which returns updated weights to the model more likely to provide convergence to local minima. One of the key parameters of ADAM is the learning rate (LR) which is set to be lr=0.001.

The training Loss at each Epoch is shown below.

![plot](https://github.com/usamanaveed900/FAISS_API/blob/main/README_Images/Training%20Loss%20vs%20epoch.PNG)

The model Accuracy on the evaluation split to compare the performance at each step of the epoch.
![plot](https://github.com/usamanaveed900/FAISS_API/blob/main/README_Images/Training%20Accuracy%20vs%20epoch.PNG)

* Accuracy after 20 EPOCH = 60%
* Accuracy after 150 EPOCH = 64%

During the 150 Epoch steps the Model hits the max Accuracy at 43 steps with an accuracy of around 66%.

#### Note
On CPU this model takes around 5 days to train for only 150 epoch.


## API Methods

### GET Status
```
import requests

host = 'http://localhost:8080'   #Localhost

# API Call and print
url =  host+'/healthcheck'
resp = requests.get(url) 
print(resp.json())
```

``` 
{'message': 'API is up and running!'}
```
### POST Image Embeding

```
import os
import requests

host = 'http://127.0.0.1:8080'   # local instance adress 

file_path = 'Datasets\\images_fb\\images\\'                  
file_name = '00a1664b-5017-4eb1-be6f-2439114505c5' # example with computer & software

# Feature extraction
url = host+'/predict/feature_embedding'
file = {'image': open(file_path+file_name+'.jpg', 'rb')}
embedings_req = requests.post(url=url,files=file)
print(embedings_req.json())
```

```  
{'features': [0.14645007252693176, 0.1447003185749054, 0.010468872264027596, 0.018565727397799492, 0.12015146017074585, 0.072574682533741, -0.11820855736732483, -0.06668830662965775, -0.04668024182319641, 0.13109275698661804, -0.09801457077264786, 0.055759914219379425, -0.06549259275197983]}
```

### For Ploting the Images In Grid with Categories

```
from IPython.display import Image as Im
from PIL import Image
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid

def plot_image_with_category(file_path,file_name,category_str):
    fig = plt.figure(figsize=(10., 10.), dpi=80)
    grid = ImageGrid(fig, 111, 
                    nrows_ncols=(2, 2), 
                    axes_pad=0.1,
                    )  
    
    img_arr = []
    for image_ID in file_name:
        img_arr.append(Image.open(file_path + image_ID + '.jpg'))
    i = 0
    for ax, im in zip(grid, img_arr):
        ax.imshow(im)
        ax.text(0, 30, 'Similar Images And Labels:'+str(category_str[i]), style='oblique',bbox={'facecolor': 'green', 'alpha': 0.75, 'pad': 10})
        i +=1
    plt.show()
```

### POST Similar Images From Base

```
import requests
import pandas as pd

    
host = 'http://127.0.0.1:8080'  

file_path = 'Datasets\\images_fb\\images\\' 

file_name = '00a1664b-5017-4eb1-be6f-2439114505c5' # example with Computer & Software


# Similar Images
url = host+'/predict/similar_images'
file = {'image': open(file_path+file_name+'.jpg', 'rb')} 
resp = requests.post(url=url,files=file)
resp_dic = dict(resp.json())
print(resp_dic)

# Getting Categories of the Similar Images from the Training Data
training_data = pd.read_csv('Datasets/training_data.csv')

image_id_to_label = {row['id_x']: row['label'] for index, row in training_data.iterrows()}
image_labels = resp_dic.get("image_labels", [])

labels = [image_id_to_label[image_id] for image_id in image_labels]

label_encoder = {"Home & Garden": 0, "Baby & Kids Stuff": 1, "DIY Tools & Materials": 2, "Music, Films, Books & Games": 3, "Phones, Mobile Phones & Telecoms": 4, "Clothes, Footwear & Accessories": 5, "Other Goods": 6, "Health & Beauty": 7, "Sports, Leisure & Travel": 8, "Appliances": 9, "Computers & Software": 10, "Office Furniture & Equipment": 11, "Video Games & Consoles": 12}
category=[]
for label in labels:
    category.append(next(key for key, value in label_encoder.items() if value == label))
print(category)

# # Plot images
plot_image_with_category(file_path,image_labels,category)
```

```
{'similar_index': [7268, 7538, 6865, 7583], 'image_labels': ['fde1c54f-dc76-4f5f-b646-138febe6c7f4', 'd20d8da3-1a85-448f-9032-a2a52e91277d', '0b57207d-462c-43f1-8653-afbe54ad9e3a', 'a074bc84-df8f-4c09-9959-77fd54372b88']}
['Computers & Software', 'Computers & Software', 'Computers & Software', 'Computers & Software']
```
After proccessing:

![plot](https://github.com/usamanaveed900/FAISS_API/blob/main/README_Images/output.png)



