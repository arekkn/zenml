import streamlit as st
import altair as alt
import pandas as pd
import numpy as np
import os
import is_cat 
from PIL import Image
import requests
from io import BytesIO
from embeddings_pipeline import run_2
@st.cache
def load_image(img_file):
    img=Image.open(img_file)
    return img
@st.cache
def find_similar(img_file):
    files=[os.path.join("Korat", f) for f in os.listdir("Korat")]
    return files

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
        if not is_cat.is_cat(load_image(img_file)):
            st.text("run start")
            run_2()
            st.text("run stop")
            st.text("And our model says he doesn't see a cat in this photo, instead he sees:")
            st.write(is_cat.what_is(load_image(img_file)))
        st.text("And our model ")
        st.text('Below are six cats similar to your cat')
        buf = BytesIO()
        col1, col2,col3= st.columns(3)
        col1.write("This is cat 1")
        img_cat1 = scale_photo(load_image(find_similar(img_file)[0]))
        col1.image(img_cat1)
        img_cat1.save(buf, format="JPEG")
        byte_im = buf.getvalue()
        col1.download_button(label='Download cat 1',data=byte_im,file_name="cat1.jpg",mime="image/jpeg")
        col2.write("This is cat 2")
        img_cat2 = scale_photo(load_image(find_similar(img_file)[1]))
        col2.image(img_cat2)
        img_cat2.save(buf, format="JPEG")
        byte_im = buf.getvalue()
        col2.download_button(label='Download cat 2',data=byte_im,file_name="cat2.jpg",mime="image/jpeg")
        col3.write("This is cat 3")
        img_cat3 = scale_photo(load_image(find_similar(img_file)[2]))
        col3.image(img_cat3)
        img_cat3.save(buf, format="JPEG")
        byte_im = buf.getvalue()
        col3.download_button(label='Download cat 3',data=byte_im,file_name="cat3.jpg",mime="image/jpeg")
        col1.write("This is cat 4")
        img_cat4 = scale_photo(load_image(find_similar(img_file)[3]))
        col1.image(img_cat4)
        img_cat4.save(buf, format="JPEG")
        byte_im = buf.getvalue()
        col1.download_button(label='Download cat 4',data=byte_im,file_name="cat4.jpg",mime="image/jpeg")
        col2.write("This is cat 5")
        img_cat5 = scale_photo(load_image(find_similar(img_file)[4]))
        col2.image(img_cat5)
        img_cat5.save(buf, format="JPEG")
        byte_im = buf.getvalue()
        col2.download_button(label='Download cat 5',data=byte_im,file_name="cat5.jpg",mime="image/jpeg")
        col3.write("This is cat 6")
        img_cat6 = scale_photo(load_image(find_similar(img_file)[5]))
        col3.image(img_cat6)
        img_cat6.save(buf, format="JPEG")
        byte_im = buf.getvalue()
        col3.download_button(label='Download cat 6',data=byte_im,file_name="cat6.jpg",mime="image/jpeg")
        st.text('Please send feedback')
        message = st.text_input('message')
        #for f in find_similar(img_file):
        #    st.image(scale_photo(load_image(f)))

    st.markdown("# End of site bye bye!")

if __name__ == '__main__':
    main()
