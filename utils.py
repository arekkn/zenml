from credentials import cat_api_key
import requests
import json
import pickle
from PIL import Image
import os


def get_list_of_cat_links(size_in_hundreds=86, backup=True):
    links = []
    for i in range(size_in_hundreds):
        links.extend(get_hundred_links(i))
    if backup:
        with open(f'{size_in_hundreds}links.pickle', 'wb') as handle:
            pickle.dump(links, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return links


def get_hundred_links(page):
    url = "https://api.thecatapi.com/v1/images/search"
    querystring = {'limit': 100, 'format': 'json', 'order': 'ASC', 'page': page}
    headers = {'x-api-key': cat_api_key}
    response = requests.request("GET", url, headers=headers, params=querystring)
    return [i['url'] for i in json.loads(response.text)]


def get_cat_from_link(link, path='../cats/'):
    cat = requests.get(link)
    with open(f'{path}{link.split("/")[-1]}', 'wb') as cat_file:
        cat_file.write(cat.content)


def get_cats(number_in_hundreds=100, backup_links=True):
    links = get_list_of_cat_links(number_in_hundreds, backup_links)
    for link in links:
        get_cat_from_link(link)


def center_crop(img):
    width, height = img.size
    new_size = min(width, height)
    xcenter = img.width // 2
    ycenter = img.height // 2
    a = new_size // 2
    x1 = xcenter - a
    y1 = ycenter - a
    x2 = xcenter + a
    y2 = ycenter + a
    img_cropped = img.crop((x1, y1, x2, y2))
    return img_cropped


def scale_photo(img):
    scaled_img = img.resize((224, 224))
    return scaled_img


def crop_and_scale(img):
    return scale_photo(center_crop(img))


def eval(cat, embedding_model, weights):
    transformer = weights.transforms()
    embedding = embedding_model(transformer(cat).unsqueeze(0)).squeeze(0).detach().numpy()
    return embedding

def eval_vit(cat, model, feature_extractor):
    encoding = feature_extractor(images=cat, return_tensors="pt")
    pixel_values = encoding['pixel_values']
    outputs = model(pixel_values, output_hidden_states=True)
    embedding = outputs.hidden_states[-1].squeeze(0)[0, :].detach().numpy()
    return embedding


if __name__ == '__main__':
    get_cats()
