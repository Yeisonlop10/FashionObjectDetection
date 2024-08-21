import streamlit as st

st.set_page_config(page_title="Home", layout='wide', page_icon="./images/home.png")
st.title("YOLO V8 Object Detection APP Fashion")
st.caption('This web app demonstrates Object detection')

# Content
st.markdown("""
### This App detects objects from Images
- Detection from 10 different classes
- [Click here for Image](/YOLO_for_Image)
- [Click here for Video](/YOLO_for_Video)
- [Click here for Real Time Video Detection](/YOLO_for_RealTime_Video)

Classes that this model detects:
            
1. Sunglass
2. Hat
3. Jacket
4. Shirt
5. Pants
6. Shorts 
7. Skirt
8. Dress
9. Bag
10. Shoe

""")