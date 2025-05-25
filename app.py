import streamlit as st
import os
import time
import pandas as pd 
import cv2
import numpy as np
import torch
from PIL import Image
from ingredients import ingredient_database  
from predict import predict  

# Base directory of this file
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def get_path(filename):
    return os.path.join(BASE_DIR, filename)

def local_css():
    css_path = get_path("style.css")
    with open(css_path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

local_css()

st.title("CareCanvas - AI-Powered Skin Analysis App")

sidebar = st.sidebar
sidebar.image(get_path("logo.jpg"), width=300)  # fixed

sidebar.markdown("----")

sidebar_option = sidebar.radio(
    "Choose an option:",
    ("Home", "Skin Type Assesment", "AI Skin Analysis", 
     "Ingredient Checker", "Personalized Skincare Solution", "Help"),
    key="sidebar_option"
)

if sidebar_option == "Home":
    st.title("🌿 Welcome to CareCanvas")

    st.markdown("""
    CareCanvas is your personal skincare companion, designed to help you understand your skin better and find the best products for your unique needs.
    """)

    st.markdown("---")

    st.header("✨ Features")

    st.subheader("📸 Facial Analysis")
    st.markdown("""
    Upload a selfie and get instant, detailed analysis of your skin concerns such as acne, dryness, spots, and more.  
    🔍 Detect multiple issues at once  
    📊 Receive severity scores  
    🧴 Get personalized skincare recommendations tailored to your unique skin.
    """)

    st.subheader("🧪 Ingredient Checker")
    st.markdown("""
    Scan or enter skincare ingredients to instantly discover whether they’re suitable for your skin type and concerns.  
    🔍 Learn the function of each ingredient  
    ⚠️ Identify potentially harmful substances  
    ✅ Explore safer alternatives for healthier skincare.
    """)

    st.subheader("🕒 Routine Generator")
    st.markdown("""
    Receive a personalized morning and evening skincare routine designed specifically for your skin type, concerns, and goals.  
    🧼 Step-by-step guidance  
    🛍️ Customized product recommendations  
    🔄 Easily adjust your routine as your skin evolves.
    """)

    st.markdown("---")
    st.success("✨ Start your journey to better skin with CareCanvas today!")

    # st.image(get_path("home.jpg"), use_container_width=True)  # fixed

elif sidebar_option == "Skin Type Assesment":
    st.subheader("Find Your Skin Type")
    st.write("Take the **Skin Analysis Quiz** to receive personalized skincare recommendations.")

    questions = [
        "1. After cleansing your face with a mild face wash, how does your skin feel after 30 minutes?",
        "2. How often does your face appear shiny or greasy throughout the day?",
        "3. How does your skin feel when exposed to cold weather or air conditioning for long hours?",
        "4. How frequently do you get acne breakouts, including pimples or cystic acne?",
        "5. If you skip your skincare routine for a day, how does your skin respond?",
        "6. How often do you experience redness, irritation, or burning sensations after using new skincare products?"
    ]

    options = [
        ['A) Tight, flaky, or rough', 'B) Comfortable, no dryness or oil', 'C) Oily and shiny', 'D) Dry in some areas, oily in others', 'E) Red or irritated'],
        ['A) Never', 'B) Occasionally', 'C) Always', 'D) Only in certain areas', 'E) My skin gets irritated instead of oily'],
        ['A) Extremely dry and flaky', 'B) Feels fine, no major changes', 'C) Becomes greasy', 'D) Some areas feel dry, some oily', 'E) Becomes red, itchy, or irritated'],
        ['A) Almost never', 'B) Occasionally during stress or hormonal changes', 'C) Frequently', 'D) Only in some areas like the chin or forehead', 'E) My breakouts are caused by irritation or allergies'],
        ['A) Feels rough and dry', 'B) No major change', 'C) Becomes oily and congested', 'D) Some areas feel dry, others oily', 'E) Becomes itchy or inflamed'],
        ['A) Almost never', 'B) Occasionally with certain products', 'C) Frequently with new skincare', 'D) Some areas react, others don’t', 'E) Almost all skincare causes reactions']
    ]

    if "quiz_answers" not in st.session_state:
        st.session_state.quiz_answers = {f"q{idx}": None for idx in range(len(questions))}

    def determine_skin_type(answers):
        skin_types = {'Dry': 0, 'Normal': 0, 'Oily': 0, 'Combination': 0, 'Sensitive': 0, 'Acne-Prone': 0}
        
        for answer in answers:
            if answer == 'A': skin_types['Dry'] += 1
            elif answer == 'B': skin_types['Normal'] += 1
            elif answer == 'C': skin_types['Oily'] += 1
            elif answer == 'D': skin_types['Combination'] += 1
            elif answer == 'E': skin_types['Sensitive'] += 1

        return max(skin_types, key=skin_types.get)

    user_answers = []
    for idx, question in enumerate(questions):
        st.subheader(question)
        answer = st.radio(
            "", options[idx], key=f"q{idx}",
            index=None if st.session_state.quiz_answers[f"q{idx}"] is None else options[idx].index(st.session_state.quiz_answers[f"q{idx}"])
        )
        if answer:
            st.session_state.quiz_answers[f"q{idx}"] = answer  
            user_answers.append(answer[0])

    col1, col2 = st.columns(2)
    with col1:
        submit_button = st.button('Submit')
    with col2:
        reset_button = st.button('Reset')

    if submit_button and len(user_answers) == len(questions):
        with st.spinner('🔍 Analyzing your skin...'):
            time.sleep(3)
            skin_type = determine_skin_type(user_answers)

        st.subheader(f"Your Skin Type: {skin_type}")

    if reset_button:
        st.session_state.quiz_answers = {f"q{idx}": None for idx in range(len(questions))}  
        st.rerun() 

elif sidebar_option == "AI Skin Analysis":
    st.subheader("Upload Your Skin Image")

    uploaded_file = st.file_uploader("Upload an image of your face", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image_path = get_path("temp_image.jpg")  # fixed
        with open(image_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        st.image(image_path, caption="Uploaded Image", use_container_width=True)

        predictions, save_path = predict(image_path)
        top_concerns = predictions[:3]  
        

        st.subheader("Predicted Skin Concerns:")
        st.write(", ".join(top_concerns) if top_concerns else "No concerns detected.")

        recommended_ingredients = []
        for ingredient, details in ingredient_database.items():
            concern_match = any(concern.lower() in [c.lower() for c in details["concerns"]] for concern in top_concerns)
            if concern_match:
                recommended_ingredients.append([
                    ingredient,
                    details["category"],
                    ", ".join(details["suitable_for"]),
                    details["benefits"],
                    details["usage"]
                ])

        if recommended_ingredients:
            df = pd.DataFrame(recommended_ingredients, 
                              columns=["Ingredient", "Category", "Suitable For", "Benefits", "Usage"])
            st.subheader("Recommended Skincare Ingredients:")
            st.dataframe(df, use_container_width=True)
        else:
            st.warning("No matching ingredients found. Try refining your concerns!")

        # Cleanup temp image
        os.remove(image_path)

elif sidebar_option == "Ingredient Checker":
    st.subheader("Check Your Skincare Ingredients")
    ingredient_input = st.text_area("Enter Ingredients (comma-separated)", "E.g., Hyaluronic Acid, Vitamin C")

    if st.button("Analyze") and ingredient_input:
        ingredient_list = [i.strip().title() for i in ingredient_input.split(",")]  
        data = []

        for ingredient in ingredient_list:
            if ingredient in ingredient_database:
                info = ingredient_database[ingredient]
                data.append([
                    ingredient,
                    info['category'],
                    ", ".join(info['suitable_for']),
                    info['benefits'],
                    info['harmful_for']
                ])
            else:
                data.append([ingredient, "Not Found", "N/A", "N/A", "N/A"])

        df = pd.DataFrame(data, columns=["Ingredient", "Category", "Suitable For", "Benefits", "Harmful For"])
        st.dataframe(df, use_container_width=True) 

elif sidebar_option == "Personalized Skincare Solution":
    st.subheader("Find the Best Skincare Ingredients for You!")

    skin_types = ["Dry", "Oily", "Normal", "Combination", "Sensitive", "Acne-Prone"]
    user_skin_type = st.selectbox("Select Your Skin Type", skin_types)

    concern_input = st.text_area("Describe Your Skin Concerns (comma-separated)", 
                                 "E.g., acne, redness, dark spots")

    if st.button("Get Recommendations"):
        concerns = [c.strip().lower() for c in concern_input.split(",")]  
        recommended_ingredients = []

        for ingredient, details in ingredient_database.items():
            matches_skin_type = user_skin_type in details["suitable_for"]

            ingredient_concerns = [c.lower() for c in details["concerns"]]
            matches_any_concern = any(concern in ingredient_concerns for concern in concerns)

            if matches_skin_type and matches_any_concern:
                recommended_ingredients.append([
                    ingredient,
                    details["category"],
                    ", ".join(details["suitable_for"]),
                    details["benefits"],
                    details["usage"]
                ])

        if recommended_ingredients:
            df = pd.DataFrame(recommended_ingredients, 
                              columns=["Ingredient", "Category", "Suitable For", "Benefits", "Usage"])
            st.subheader("Recommended Ingredients for Your Skin Type & Concerns:")
            st.dataframe(df, use_container_width=True)
        else:
            st.warning("No matching ingredients found. Try refining your concerns!")

elif sidebar_option == "Help":
    st.subheader("Frequently Asked Questions")

    faq_data = {
        "What kind of images should I upload for analysis?": 
        "Ensure good lighting and avoid wearing makeup. The image should be clear, front-facing, and display your natural skin texture.",

        "How accurate is the AI analysis?": 
        "Our AI model is trained on a large dataset of dermatological images. While it provides reliable results, it is not a substitute for professional medical advice. For serious skin conditions, consulting a dermatologist is recommended.",

        "Can I use this app if I have sensitive skin?": 
        "Yes. SkinSage provides ingredient recommendations that are suitable for sensitive skin. You can also use the Ingredient Checker to verify product compatibility.",

        "Does this app replace a dermatologist?": 
        "No. SkinSage is an AI-powered tool designed to assist with skincare decisions but should not be used as a replacement for professional dermatological consultations.",

        "How often should I analyze my skin?": 
        "A monthly analysis is recommended to track changes in skin condition and adjust your skincare routine accordingly.",

        "Is my data safe?": 
        "Yes. SkinSage does not store uploaded images, and all processing occurs in real time. No personal data is shared with third parties."
    }

    for question, answer in faq_data.items():
        with st.expander(question):
            st.write(answer)

    st.markdown("For further inquiries, please contact us at **supportskinsage@com**.")

st.markdown('<p class="footer">© 2025 SkinSage</p>', unsafe_allow_html=True)
