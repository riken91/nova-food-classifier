import streamlit as st
import joblib
import numpy as np

# Load model
model = joblib.load("best_random_forest_model.pkl")
feature_names = model.feature_names_in_

# Define group → emoji mapping
GROUP_EMOJIS = {
    "water": "💧", "salt_minerals": "🧂", "sugars": "🍯", "sweeteners": "🍬",
    "dairy": "🥛", "animal_meat": "🍖", "fish_seafood": "🐟", "egg_products": "🥚",
    "plant_proteins": "🌱", "nuts_seeds": "🥜", "grains_cereals": "🌾", "starches": "🍠",
    "fats_oils": "🛢️", "fruits": "🍎", "vegetables": "🥦", "herbs_spices": "🌿",
    "flavourings": "🍫", "thickeners_gelling": "🍮", "leavening_agents": "🥯",
    "preservatives": "💊", "emulsifiers": "🧴", "stabilizers": "🧊",
    "acidity_regulators": "🍋", "antioxidants": "🛡️", "colourants": "🎨",
    "vitamins_minerals": "💊", "fermented_cultures": "🥒", "beverage_ingredients": "☕",
    "condiments_sauces": "🥫", "seaweed_algae": "🌊", "coating_glazing": "🍬",
    "processing_aids": "⚙️", "functional_additions": "➕", "mixed_composites": "🍲",
    "packaging_traces": "📦", "unclassified_misc": "❓", "numeric_artifacts": "🔢"
}

# Streamlit app
st.title("🍴 Predict the NOVA Group (1 to 4)")
st.write("Based on nutritional values and ingredients")

# Input box
ingredients_text = st.text_area(
    "Enter ingredients (comma-separated):",
    "honey, small red beans, stabilizers"
)

if st.button("Predict"):
    # Clean user input
    user_ingredients = [ing.strip().lower() for ing in ingredients_text.split(",") if ing.strip()]
    
    # Map to groups (simple keyword matching)
    matched_groups = []
    for ing in user_ingredients:
        for group in feature_names:
            if group in ing:  # crude match: can be improved with GROUP_KEYWORDS
                matched_groups.append(group)
                break

    # Create feature vector
    vec = np.array([1 if feat in matched_groups else 0 for feat in feature_names]).reshape(1, -1)

    # Predict with model
    pred_class = model.predict(vec)[0]
    pred_proba = model.predict_proba(vec)[0]

    # Toggle option: Show original vs mapped groups
    view_option = st.radio(
        "How do you want to view your inputs?",
        ("Show original ingredients", "Show mapped groups")
    )

    if view_option == "Show original ingredients":
        st.write("✅ Ingredients you entered:")
        st.write(", ".join(user_ingredients))
    else:
        st.write("✅ Mapped Groups:")
        pretty_groups = [f"{GROUP_EMOJIS.get(g, '')} {g}" for g in matched_groups]
        st.write(", ".join(pretty_groups))

    # Show prediction
    st.success(f"Predicted NOVA Group: {int(pred_class) + 1}")  # shift 0→1, 1→2 etc.
    
    st.write("📊 Class Probabilities")
    for cls, prob in zip(model.classes_, pred_proba):
        st.write(f"Class {int(cls) + 1}: {prob:.2f}")
