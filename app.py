import streamlit as st
import joblib
import numpy as np

# Load model
model = joblib.load("best_random_forest_model.pkl")
feature_names = model.feature_names_in_

# Ingredient groups (37 categories)
ingredient_groups = [
    "dairy", "fermented_cultures", "sugars", "fats_oils", "flavourings",
    "additives", "cereals", "fruits", "vegetables", "nuts", "legumes",
    "protein_sources", "functional_additions", "minerals", "vitamins",
    "spices", "starches", "emulsifiers", "preservatives", "acids",
    "sweeteners", "colorants", "fibres", "animal_products", "yeast",
    "alcohol", "water", "salt", "herbs", "seafood", "seeds",
    "flour", "gums", "coffee_cocoa", "tea", "miscellaneous", "other"
]

# Dummy ingredient list (replace with your real list later)
all_ingredients = [
    "milk", "yoghurt", "cream", "cheddar cheese", "sugar", "honey",
    "olive oil", "butter", "almonds", "peanuts", "tomato", "potato",
    "wheat flour", "chicken", "beef", "egg", "salt", "pepper",
    "vinegar", "vanilla extract"
]

st.title("🍴 Predict the NOVA Group (1 to 4)")
st.subheader("Based on nutritional values and ingredients")

# Option to choose groups vs ingredients
option = st.radio("Select input mode:", ["Ingredients", "Groups"])

if option == "Ingredients":
    selected_items = st.multiselect(
        "🛒 Pick ingredients:",
        all_ingredients
    )
else:
    selected_items = st.multiselect(
        "📦 Pick ingredient groups:",
        ingredient_groups
    )

if st.button("Predict"):
    user_inputs = [ing.lower() for ing in selected_items]
    vec = np.array([1 if feat in user_inputs else 0 for feat in feature_names]).reshape(1, -1)
    
    pred_class = model.predict(vec)[0]
    pred_proba = model.predict_proba(vec)[0]

    st.success(f"Predicted NOVA Group: {int(pred_class)+1}")  # +1 if model is 0-3
    st.write("📊 Class Probabilities")
    for cls, prob in zip(model.classes_, pred_proba):
        st.write(f"Class {int(cls)+1}: {prob:.2f}")
