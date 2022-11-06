
# Find your cat a soulmate!

## An entry for ZenML month of MLOps competition

### Problem statement

As a team of beginners, we wanted to do something fun, because of lack of experience dealing with real-world MLOps problems. Given a photo from the user, we want to find the most similar cats in our "database". Additionally, if the photo contains a cat, we want to add it to our data.


### Data

For our cat dataset, we used https://thecatapi.com/. We downloaded 9000 cat photos and gifs. We filtered them by image format, deleting gifs, and then cropped them to universal deep learning format of 224x224 pixels. You can see that in the pipeline download_data, but to run this part, you need to add file `credentials.py` in the main folder containing one line - `cat_api_key = "<your api key>"`, as we didn't want to put our api key online. You can get the api key with free registration on the site mentioned above.

### Model to validate the data (check if a photo contains a cat)

For this we used Resnet50 pretrained on imagenet. If Resnet predicted any of the 'catlike' classes with probability (after softmax) of at least 0.5%, we classify it as a cat. We found this threshold to be satisfactory.

  

### Models to find similar cat

Every model mentioned below has the same idea behind it: find some kind of embeddings for the cats we downloaded, and as the user sends the picture of his cat, calculate the embedding for his picture, and then find the closest embedding in our database. You can generate the embeddings for all the photos by running the pipeline `find_embeddings`.

  

#### Autoencoder

Our first idea was to create an autoencoder that we would train ourselves to find "the essence of cat" on a cat photo, however it turned out to not be so easy. First of all our, autoencoder did not train really well. We believe it was because the dataset was a bit to small, and our model was a bit too shallow, as decoded images looked like the input images, but very compressed. Our model did not learb how to diffrentiate between cats, but how to diffrentiate between background photos. You can see this by running `train_autoencoder` pipeline, that trains our model incrementally, but it needs a saved model, which we couldn't push to GitHub due to its size.

#### Resnet based Models

We used a few Resnet based models pretrained on ImageNet. Our embeddings are the outputs of the last layer before the final classification layer (the one that outputs 2048x1 vector). It turned out to work okay, especially when the picture contained some kind of characteristic thing other than the cat itself. For example, for a picture of a cat with glasses, those would likely return a picture of a cat with glasses.

#### Vision Transformer

The model that worked best for our task was a vision transformer (pretrained on ImageNet). We used it in a manner similar to the Resnet architectures, meaning we used the outputs of the final layer before the dense classification layer. The result is a vector with the size of 768z1. It seems like it was the best performing model, because unlike CNNs, the transformer "understands" the images an a global level.

  
  


  

### Frontend app

We used Streamlit for the demo app. It is free and allows for a simple presentation of what we accomplished during the ZenML month of MLOps competition.[Streamlit-app](https://arekkn-zenml-streamlit-app-htd0eo.streamlit.app/)

### Authors
- [Arkadiusz Kniaź](https://github.com/arekkn)
- [Hubert Kozubek](https://github.com/HKozubek)
- [Jan Kruszewski](https://github.com/Janekkr)
- [Andrzej Pióro](https://github.com/AndrzejMnM)
- [Grzegorz Kiersnowski](https://github.com/Grzgorzk)
