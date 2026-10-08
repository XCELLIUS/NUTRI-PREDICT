# NutriPred 🥗

AI-assisted nutrition recommendation and meal-planning system built using Python, Scikit-learn, Pandas, and Streamlit.

WEBSITE:- https://nutri-predict-main.vercel.app/

## Features

- Patient profile input
- BMI calculation
- Random Forest diet recommendation
- Diet recommendation probabilities
- Nutrition-aware food selection
- Allergy filtering
- Daily calorie targeting
- Breakfast, lunch, snack, and dinner planning
- Portion-size calculation
- Protein, carbohydrate, fat, fiber, and sodium analysis
- Interactive Streamlit dashboard
- Meal-plan CSV export

## Machine Learning

### Diet Recommendation Model

Algorithm:

Random Forest Classifier

Classes:

- Balanced
- Low_Carb
- Low_Sodium

The model uses patient demographic, anthropometric, metabolic, lifestyle, dietary, and preference features.

### Metabolic Model

Algorithm:

Random Forest Regressor

The model uses 14 numerical features.

## Food Database

The application uses a processed nutritional food database containing:

- Calories
- Protein
- Fat
- Carbohydrates
- Fiber
- Sodium

## Application Architecture

```text
Patient Input
     ↓
BMI Calculation
     ↓
Diet Recommendation Model
     ↓
Diet Category
     ↓
Food Database
     ↓
Allergy Filtering
     ↓
Nutrition Scoring
     ↓
Calorie/Portion Optimization
     ↓
Breakfast / Lunch / Snack / Dinner
     ↓
Nutrition Analysis
     ↓
Streamlit Dashboard
