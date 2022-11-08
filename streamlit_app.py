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
from utils import crop_and_scale
from transformers import ViTFeatureExtractor
from transformers import ViTForImageClassification
import torchvision.models

@st.cache(max_entries=1)
def load_image(img_file):
    img=Image.open(img_file)
    return img
 
###dodaje   
@st.cache(max_entries=1)
def find_closest_img(reasult):
    df = pd.read_pickle("./embeddings.pkl", compression="xz") 
    our_cat = reasult
    result_img_list = []
    for col in df.columns[1:]:
        d = np.stack((df[col].to_numpy() - our_cat[col].to_numpy())**2).sum(axis=1)
        d[d.argmin()] = float("inf")
        idx = d.argmin()
        result_img_list.append(df.iloc[idx]["img_paths"])
    del df
    return result_img_list

@st.cache(max_entries=1)
def save_new_embedings(img_path):
    img= crop_and_scale(Image.open(img_path))
    # embeddings_df = pd.read_pickle("embeddings.pkl",compression="xz")
    resnet18_weights = torchvision.models.ResNet18_Weights.DEFAULT
    resnet18_model = torchvision.models.resnet18(weights=resnet18_weights)
    resnet34_weights = torchvision.models.ResNet34_Weights.DEFAULT
    resnet34_model = torchvision.models.resnet34(weights=resnet34_weights)
    resnet50_weights = torchvision.models.ResNet50_Weights.DEFAULT
    resnet50_model = torchvision.models.resnet50(weights=resnet50_weights)
    feature_extractor = ViTFeatureExtractor('google/vit-base-patch16-224')
    embedding_model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')
    reasult = pd.DataFrame({
        "img_paths":[img_path],
        "resnet18_embeddings":[eval(img, resnet18_model, resnet18_weights)],
        "resnet34_embeddings":[eval(img, resnet34_model, resnet34_weights)],
        "resnet50_embeddings":[eval(img, resnet50_model, resnet50_weights)],
        "vit_embeddings": [eval_vit(img, embedding_model, feature_extractor)]})
    return reasult


@st.cache(max_entries=1)
def scale_photo(img):
    scaled_img = img.resize((224, 224))
    return scaled_img

def main():

    st.markdown("# Welcome to Find your soulcat!")
    st.text("Upload your cat picture and see the magic!")
    img_file = st.file_uploader(label=" ",type=["png","jpg","jpeg"],label_visibility="hidden")
    if img_file is not None:
        st.text("So this is your picture.")

        st.image(load_image(img_file))

        cat = is_cat.is_cat(load_image(img_file))

        img_path=os.path.join("cats/", img_file.name)
        with open(img_path, "wb") as f:
            f.write(img_file.getbuffer())
        #
        reasult=save_new_embedings(img_path)
        #
        recomendation_list = find_closest_img(reasult)
        #
        if not cat:
            st.text("And our model says he doesn't see a cat in this picture, instead he sees:")
            not_cat=is_cat.what_is(load_image(img_file))
            st.write(f"{not_cat.iloc[0,2]} with {100 *not_cat.iloc[0,1]:.1f}% certainty.")
        else:
            st.text("And our model sees cat on your picture.")
        st.text('Below are four cats similar to your picture.')
        
        col1, col2,col3= st.columns(3)

        col1.write("This is cat 1")
        img_cat1 = scale_photo(load_image(recomendation_list[0]))
        col1.image(img_cat1)
        buf1 = BytesIO()
        img_cat1.save(buf1, format="JPEG")
        byte_im1 = buf1.getvalue()
        col1.download_button(label='Download cat 1',data=byte_im1,file_name="cat1.jpg",mime="image/jpeg")

        col2.write("This is cat 2")
        img_cat2 = scale_photo(load_image(recomendation_list[1]))
        col2.image(img_cat2)
        buf2 = BytesIO()
        img_cat2.save(buf2, format="JPEG")
        byte_im2= buf2.getvalue()
        col2.download_button(label='Download cat 2',data=byte_im2,file_name="cat2.jpg",mime="image/jpeg")


        col3.write("This is cat 3")
        img_cat3 = scale_photo(load_image(recomendation_list[2]))
        col3.image(img_cat3)
        buf3 = BytesIO()
        img_cat3.save(buf3, format="JPEG")
        byte_im3 = buf3.getvalue()
        col3.download_button(label='Download cat 3',data=byte_im3,file_name="cat3.jpg",mime="image/jpeg")

        col1.write("This is cat 4")
        img_cat4 = scale_photo(load_image(recomendation_list[3]))
        col1.image(img_cat4)
        buf4 = BytesIO()
        img_cat4.save(buf4, format="JPEG")
        byte_im4 = buf4.getvalue()
        col1.download_button(label='Download cat 4',data=byte_im4,file_name="cat4.jpg",mime="image/jpeg")
        

        st.markdown("## WOW!🤩😲🤩")
        st.write("Now try with another cat or maybe with something else, maybe even your selfie.")

    st.markdown("### Site by WUT's students for ZenML's Month of MLOps.")
    st.markdown("### See our full project on [github](https://github.com/arekkn/zenml)")

if __name__ == '__main__':
    main()
