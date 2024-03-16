import streamlit as st
from keras.models import load_model
from keras.preprocessing import image
import numpy as np
from PIL import Image, ImageOps

st.set_page_config(
    page_title="Demo",
    page_icon="🧊",
    layout="wide",
    initial_sidebar_state="expanded",
)

def predict(input_image):
    labels = ["Hiragana", "Kanji", "Katakana"]
    # load mô hình phân loại
    model = load_model("./models/M2.h5")
    
    # xử lý ảnh
    size = (100, 100)
    img = ImageOps.fit(input_image, size)
    img_tensor = image.img_to_array(img)
    img_tensor = np.expand_dims(img_tensor, axis=0)
    img_tensor /= 255
    
    # thực hiện phân loại & trả kết quả
    pred = model.predict(img_tensor)
    return labels[np.argmax(pred[0])]

def main():
    st.markdown("<h1 style='text-align: center; color: black;'>Phân loại ký tự tiếng Nhật</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: grey; font-size: 18px;'>Demo phân loại ký tự tiếng Nhật theo 3 hệ thống chữ viết bao gồm Hiragana, Kanji và Katakana</p>", unsafe_allow_html=True)
    # upload ảnh
    st.subheader("Chọn ảnh")
    file = st.file_uploader("", type=["jpg", "png"])
    if file is not None:
        image = Image.open(file)
        with st.columns(3)[1]:
            st.image(image)
            result = predict(image)
            st.text(f"Ký tự trên thuộc hệ thống chữ {result}")

if __name__ == "__main__":
    main()