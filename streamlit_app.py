import streamlit as st
import altair as alt
import pandas as pd
import numpy as np
import os
import is_cat 
from PIL import Image
import requests
from io import BytesIO
####dodaje
from utils import eval
from utils import eval_vit
from transformers import ViTFeatureExtractor
from transformers import ViTForImageClassification
import torchvision.models

@st.cache
def load_image(img_file):
    img=Image.open(img_file)
    return img
 
###dodaje   
@st.cache
def find_closest_img(img):
    df = pd.read_pickle("./embeddings.pkl", compression="xz") 
    our_cat = df.loc[df["img_paths"]==img]
    result_img_list = []
    for col in df.columns[1:]:
        d = np.stack((df[col].to_numpy() - our_cat[col].to_numpy())**2).sum(axis=1)
        d[d==0] = 1000
        idx = d.argmin()
        result_img_list.append(df.iloc[idx]["img_paths"])
    return result_img_list

@st.cache
def save_new_embedings(img_path):
    embeddings_df = pd.read_pickle("embeddings.pkl",compression="xz")
    resnet18_weights = torchvision.models.ResNet18_Weights.DEFAULT
    resnet18_model = torchvision.models.resnet18(weights=resnet18_weights)
    resnet34_weights = torchvision.models.ResNet34_Weights.DEFAULT
    resnet34_model = torchvision.models.resnet34(weights=resnet34_weights)
    resnet50_weights = torchvision.models.ResNet50_Weights.DEFAULT
    resnet50_model = torchvision.models.resnet50(weights=resnet50_weights)
    feature_extractor = ViTFeatureExtractor('google/vit-base-patch16-224')
    embedding_model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')
    result = pd.DataFrame({
        "img_paths":[img_path],
        "resnet18_embeddings":[eval(Image.open(img_path), resnet18_model, resnet18_weights)],
        "resnet34_embeddings":[eval(Image.open(img_path), resnet34_model, resnet34_weights)],
        "resnet50_embeddings":[eval(Image.open(img_path), resnet50_model, resnet50_weights)],
        "vit_embeddings": [eval_vit(Image.open(img_path), embedding_model, feature_extractor)]})
    output = pd.concat([embeddings_df, result])
    output.to_pickle("./embeddings.pkl")


@st.cache
def scale_photo(img):
    scaled_img = img.resize((224, 224))
    return scaled_img

def main():

    st.markdown("# Welcome to Find your soul cat!")
    st.text("Upload your cat picture and see the magic")
    img_file = st.file_uploader("",type=["png","jpg","jpeg"])
    if img_file is not None:
        st.text("So this is your picture")
        st.image(load_image(img_file))
        st.write(is_cat.is_cat(load_image(img_file)))
        #####dodaje
        img_path=os.path.join("cats/", img_file.name)
        with open(img_path, "wb") as f:
            f.write(img_file.getbuffer())
        save_new_embedings(img_path)
        recomendation_list = find_closest_img(img_path)

        if not is_cat.is_cat(load_image(img_file)):
            st.text("And our model says he doesn't see a cat in this photo, instead he sees:")
            st.write(is_cat.what_is(load_image(img_file)))
        st.text("And our model ")
        st.text('Below are six cats similar to your cat')
        buf = BytesIO()
        col1, col2,col3= st.columns(3)
        col1.write("This is cat 1")
        img_cat1 = scale_photo(load_image(recomendation_list[0]))
        col1.image(img_cat1)
        img_cat1.save(buf, format="JPEG")
        byte_im = buf.getvalue()
        col1.download_button(label='Download cat 1',data=byte_im,file_name="cat1.jpg",mime="image/jpeg")
        col2.write("This is cat 2")
        img_cat2 = scale_photo(load_image(recomendation_list[1]))
        col2.image(img_cat2)
        img_cat2.save(buf, format="JPEG")
        byte_im = buf.getvalue()
        col2.download_button(label='Download cat 2',data=byte_im,file_name="cat2.jpg",mime="image/jpeg")
        col3.write("This is cat 3")
        img_cat3 = scale_photo(load_image(recomendation_list[2]))
        col3.image(img_cat3)
        img_cat3.save(buf, format="JPEG")
        byte_im = buf.getvalue()
        col3.download_button(label='Download cat 3',data=byte_im,file_name="cat3.jpg",mime="image/jpeg")
        col1.write("This is cat 4")
        img_cat4 = scale_photo(load_image(recomendation_list[3]))
        col1.image(img_cat4)
        img_cat4.save(buf, format="JPEG")
        byte_im = buf.getvalue()
        col1.download_button(label='Download cat 4',data=byte_im,file_name="cat4.jpg",mime="image/jpeg")
        
        st.text('Please send feedback')
        message = st.text_input('message')
        

    st.markdown("# End of site bye bye!")

if __name__ == '__main__':
    main()
