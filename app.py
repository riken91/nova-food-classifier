import streamlit as st
import joblib
import numpy as np

# ---------- 1) Group dictionary ----------
GROUP_KEYWORDS = {
    "water": ["water", "aqua"],
    "salt_minerals": ["salt", "sodium chloride", "iodized salt", "sea salt", "rock salt", "potassium chloride", "sodium"],
    "sugars": ["sugar", "brown sugar", "cane sugar", "fructose", "glucose", "sucrose", "dextrose", "invert sugar", "high fructose corn syrup", "corn syrup", "maltose",
               "hfcs","honey","organic honey*","miels de fleurs de france","miel","pure maple syrup","maple syrup"],
    "sweeteners": ["stevia", "aspartame", "sucralose", "acesulfame", "acesulfame k", "saccharin", "xylitol", "erythritol", "sorbitol", "mannitol", "allulose", "neotame", "thaumatin", "monk fruit", "maltitol"],
    "dairy": ["milk", "butter", "cream", "yogurt", "yoghurt", "cheese", "whey", "casein", "ghee", "lactose", "skim milk", "whole milk", "milk powder", "buttermilk"],
    "animal_meat": ["beef", "chicken", "pork", "lamb", "meat", "turkey", "duck", "mutton"],
    "fish_seafood": ["fish", "tuna", "shrimp", "anchovy", "crab", "salmon", "prawn", "cod", "sardine", "mackerel", "shellfish"],
    "egg_products": ["egg yolk", "egg white", "albumen", "egg"],
    "plant_proteins": ["soy", "tofu", "pea protein", "chickpea", "lentil", "soybean", "tempeh", "seitan", "gluten", "textured vegetable protein", "tvp", "faba", "broad bean", "mung bean"],
    "nuts_seeds": ["almond", "cashew", "sesame", "peanut", "walnut", "hazelnut", "pistachio", "macadamia", "chia", "flax", "sunflower seed", "pumpkin seed", "pecan", "nut","dry roasted almonds"],
    "grains_cereals": ["wheat", "rice", "oats", "barley", "corn", "maize", "rye", "quinoa", "sorghum", "millet", "semolina", "spelt"],
    "starches": ["starch", "potato starch", "corn starch", "tapioca", "modified starch", "dextrin", "pre-gelatinized starch"],
    "fats_oils": ["oil", "palm", "sunflower", "olive", "canola", "coconut", "rapeseed", "vegetable oil", "shortening", "margarine"],
    "fruits": ["apple", "banana", "berry", "berries", "mango", "citrus", "orange", "lemon", "lime", "grape", "pineapple", "strawberry", "blueberry", "raspberry", "pear", "peach", "apricot", "fruit"],
    "vegetables": ["tomato", "spinach", "carrot", "onion", "garlic", "pepper", "broccoli", "cabbage", "potato", "pea", "capsicum", "celery", "vegetable"],
    "herbs_spices": ["cinnamon", "basil", "turmeric", "cumin", "pepper", "clove", "cardamom", "oregano", "thyme", "rosemary", "ginger", "spice", "herb", "chili", "chilli", "paprika"],
    "flavourings": ["vanilla", "cocoa", "coffee", "flavour", "flavor", "natural flavour", "artificial flavour", "aroma", "flavouring", "flavoring"],
    "thickeners_gelling": ["pectin", "gelatin", "agar", "carrageenan"],
    "leavening_agents": ["baking powder", "yeast", "baking soda", "sodium bicarbonate", "raising agent", "leaven"],
    "preservatives": ["benzoate", "sorbate", "nitrate", "nitrite", "sulfite", "sulphite", "propionate", "preservative"],
    "emulsifiers": ["lecithin", "mono- and diglycerides", "mono/diglycerides", "monoglyceride", "diglyceride", "polysorbate", "emulsifier", "e471"],
    "stabilizers": ["xanthan gum", "guar gum", "alginate", "locust bean gum", "gellan", "stabiliser", "stabilizer", "cellulose gum"],
    "acidity_regulators": ["citric acid", "lactic acid", "acetic acid", "malic acid", "phosphoric acid", "acid regulator", "acidulant"],
    "antioxidants": ["ascorbic acid", "tocopherol", "bha", "bht", "antioxidant"],
    "colourants": ["colour", "color", "caramel color", "annatto", "beta-carotene", "paprika extract", "e1"],
    "vitamins_minerals": ["vitamin", "iron", "calcium", "zinc", "niacin", "riboflavin", "thiamin", "thiamine", "folic", "iodine"],
    "fermented_cultures": ["starter culture", "probiotic", "culture", "lactobacillus", "bifidobacterium"],
    "beverage_ingredients": ["tea", "coffee", "cocoa", "malt extract", "malt"],
    "condiments_sauces": ["ketchup", "soy sauce", "mustard", "mayonnaise", "vinegar", "sauce"],
    "seaweed_algae": ["spirulina", "kelp", "nori", "seaweed", "algae"],
    "coating_glazing": ["beeswax", "carnauba wax", "shellac", "glazing agent", "coating"],
    "processing_aids": ["enzyme", "anti-caking", "anticaking", "silicon dioxide", "processing aid"],
    "functional_additions": ["omega", "probiotic", "fiber", "fibre", "inulin"],
    "mixed_composites": ["curry powder", "spice mix", "seasoning", "compound ingredient", "bouillon", "stock", "masala"],
    "packaging_traces": ["may contain", "traces", "allergen", "gluten"],
    "unclassified_misc": [],
    "numeric_artifacts": []
}

# ---------- 2) Load model ----------
model = joblib.load("best_random_forest_model.pkl")
feature_names = model.feature_names_in_

# ---------- 3) Build lists ----------
all_groups = list(GROUP_KEYWORDS.keys())
all_ingredients = sorted({ing.lower() for lst in GROUP_KEYWORDS.values() for ing in lst})

# ---------- 4) Streamlit UI ----------
st.title("🍴 Predict the NOVA Group (1 to 4)")
st.subheader("Based on nutritional values and ingredients")

mode = st.radio("Choose input mode:", ["Ingredients", "Groups"])

if mode == "Ingredients":
    selected_items = st.multiselect("🛒 Select ingredients:", all_ingredients)
else:
    selected_items = st.multiselect("📦 Select ingredient groups:", all_groups)

# ---------- 5) Prediction ----------
if st.button("Predict"):
    user_inputs = [x.lower() for x in selected_items]

    # Map ingredient → group if Ingredients mode is chosen
    if mode == "Ingredients":
        mapped_groups = []
        for group, items in GROUP_KEYWORDS.items():
            for ing in user_inputs:
                if ing in [i.lower() for i in items]:
                    mapped_groups.append(group)
        final_inputs = mapped_groups
    else:
        final_inputs = user_inputs

    # Convert to feature vector
    vec = np.array([1 if feat in final_inputs else 0 for feat in feature_names]).reshape(1, -1)

    pred_class = model.predict(vec)[0]
    pred_proba = model.predict_proba(vec)[0]

    st.success(f"Predicted NOVA Group: {int(pred_class)+1}")
    st.write("📊 Class Probabilities")
    for cls, prob in zip(model.classes_, pred_proba):
        st.write(f"Class {int(cls)+1}: {prob:.2f}")
