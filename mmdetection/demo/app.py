import streamlit as st
from local_components import card_container, card_container_border, ch1, ch2, ch3, ch4
from streamlit_extras.tags import tagger_component
import io
import os
import cv2
import random
import numpy as np
from PIL import Image
from argparse import ArgumentParser
import sqlite3 
import pickle

st.set_page_config(page_title="Mask Fashion", page_icon=":kimono:", layout="centered", initial_sidebar_state="auto", menu_items=None)

all_colors = ['red', 'green', 'blue', 'purple', 'orange', 'yellow', 'gray', 'lightblue', 'maroon', 'navy', 'purple', 'fuchsia', 'olive', 'teal', 'aqua', 'cyan', 'tomato', 'lime', 'orange', 'gold']

if "model_init" not in st.session_state.keys() and "model_det" not in st.session_state.keys():
    st.session_state["model_init"] = 'model_initialized' #loadPretrainedModel(embed_size, loss_type)
    from mmdet.apis import  inference_detector,init_detector
    st.session_state["model_det"] = inference_detector,init_detector
else:
    inference_detector,init_detector = st.session_state["model_det"]
if "submitted" not in st.session_state.keys():
    st.session_state.submitted = False

#loading configs and weights
def load_model():
    config_file='configs/fashionformer/fashionpedia/fashionformer_r101_mlvl_feat_8x.py'
    checkpoint_file='../data/model_final.pth'
    model=init_detector(config_file, checkpoint_file, device="cpu")
    return model

def search_products(out_det):
    # Connect to SQLite database
    _conn = sqlite3.connect('db.sqlite3')
    _cur = _conn.cursor()
    _cur.execute('SELECT * FROM product_data')
    db_results = _cur.fetchall()
    if not db_results:
        return "No data available at the moment! Try again later"
    else:
        # Calculate rating and select highest rated images
        selected_results = calculate_rating(out_det, db_results)
        return selected_results
    _conn.close()
   
def calculate_rating(new_detection, db_results):
    ratings = []
    for db_row in db_results:
        if not db_row:
            return "No data"
        img_det = pickle.loads(db_row[2])
        rating = 0
        for i, db_obj in enumerate(img_det):
            for j, obj in enumerate(new_detection):
                category = obj[0].split('|')[0]
                if category == db_obj[0]:  # Match category
                    if i == 0 and j == 0:
                        rating += 10
                    else:
                        rating += 5
                if obj[1] == db_obj[1]:  # Match super category
                    rating += 2
                    for attribute in obj[2].split('\n'):
                        if attribute.strip() in db_obj[2]:
                            rating += 0.5
                if obj[3] == db_obj[3]:  # Match color
                    rating += 1.5
        ratings.append(((rating/len(img_det))+(len(img_det)*2), [db_row[1], img_det, db_row[3]]))
    
    ratings.sort(reverse=True)
    
    selected_results = []
    for rating, obj_all in ratings:
        if rating > 5*len(new_detection):
            selected_results.append(obj_all)
        if len(selected_results) >= 6:
            break
    
    return selected_results

def infer_model(model, img):
    result = inference_detector(model, img)
    ret = model._show_result(img, result, score_thr=0.5, out_file=None)
    return ret

def main():
    if "model_saved" not in st.session_state.keys():
        model=load_model()
        st.session_state['model_saved'] = model
    else:
        model = st.session_state['model_saved']
    with st.container():
        st.header("Fashion Analysis and Cataloging Using Mask-RCNN")
        
        file_upload = st.file_uploader("Upload Image:", type=["png", "jpg", "jpeg"])

        if file_upload is not None and model:
            img = np.asarray(Image.open(io.BytesIO(file_upload.getvalue())))
            cv_img = 'image.jpg'
            img = cv2.imdecode(np.frombuffer(file_upload.getvalue(), np.uint8), cv2.IMREAD_COLOR)
            if img.shape[-1] == 4:  # Check for alpha channel
                img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
            st.image(img, channels="BGR")  # Specify BGR channel order for display
            st.write("Processing image...")
            out_img, out_det = infer_model(model, img)
            st.image(out_img, channels="BGR")
            for k, itm in enumerate(out_det):
                det = itm[2].split('\n')
                titl = itm[0].replace('|', ' - ').capitalize()
                with card_container(key=f"card{k}"):
                    ch4(text=f"Category: {itm[1].capitalize()}  |  Color: {itm[3].capitalize()}", weight="normal")
                    ch2(text=titl, weight="bolder") 
                    tagger_component("Attributes: ", det[:-1], color_name= [random.choice(all_colors) for _ in det[:-1]])
            st_btn = st.button(label="Search for Products", on_click=lambda: st.session_state.update(submitted=True))
            if st.session_state.submitted and model:
                st.subheader("Related Results")
                all_results = search_products(out_det)
                if all_results is not None:
                    if isinstance(all_results, str):
                        st.write(all_results)
                    else:
                        st.write(f"{len(all_results)} Related Results Found")
                        r_cols = 2
                        image_dir = "../all_batch"
                        for j in range(0, len(all_results), r_cols):
                            col1, col2 = st.columns(r_cols)
                            for k in range(r_cols):
                                redz = j+k
                                if redz >= len(all_results):
                                    break
                                rez = all_results[redz]
                                if rez is None:
                                    break
                                if not os.path.isfile(os.path.join(image_dir, rez[2])):
                                    print("no file oooo")
                                    break
                                if k == 0:
                                    with col1:
                                        with card_container_border(key=f"card1"):
                                            image_file = os.path.join(image_dir, rez[2])
                                            image_det = ''
                                            img_seen = set()
                                            for i, itm in enumerate(rez[1]):
                                                if itm[0] not in img_seen:
                                                    image_det += f" {itm[0]},"
                                                img_seen.add(itm[0])
                                            t_obj = rez[1][0][2].split('\n')[0]
                                            image_det += f" {t_obj}, {rez[1][0][3]}"
                                            st.image(image_file, channels="BGR")
                                            ch4(text=rez[0], weight="bolder")
                                            st.write(image_det)
                                else:
                                    with col2:
                                        with card_container_border(key=f"card2"):
                                            image_file = os.path.join(image_dir, rez[2])
                                            image_det = ''
                                            img_seen = set()
                                            for i, itm in enumerate(rez[1]):
                                                if itm[0] not in img_seen:
                                                    image_det += f" {itm[0]},"
                                                img_seen.add(itm[0])
                                            t_obj = rez[1][0][2].split('\n')[0]
                                            image_det += f" {t_obj}, {rez[1][0][3]}"           
                                            st.image(image_file, channels="BGR")
                                            ch4(text=rez[0], weight="bolder")
                                            st.write(image_det)
                else:
                    st.write("Oops! Something went wrong, refresh the page and try again. If issue persists, contact owner of the project")
                
if __name__=='__main__':
    main()