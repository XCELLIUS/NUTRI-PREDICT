import os, re, joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title='NutriPred', page_icon='🥗', layout='wide')
APP_DIR=os.path.dirname(os.path.abspath(__file__))
DIET_MODEL_PATH=os.path.join(APP_DIR,'diet_recommendation_model.pkl')
METABOLIC_MODEL_PATH=os.path.join(APP_DIR,'metabolic_glucose_model.pkl')
FOOD_PATH=os.path.join(APP_DIR,'processed_food_nutrition.csv')
DIET_FEATURES=['Age','Gender','Weight_kg','Height_cm','BMI','Disease_Type','Severity','Physical_Activity_Level','Daily_Caloric_Intake','Cholesterol_mg/dL','Blood_Pressure_mmHg','Glucose_mg/dL','Dietary_Restrictions','Allergies','Preferred_Cuisine','Weekly_Exercise_Hours','Adherence_to_Diet_Plan','Dietary_Nutrient_Imbalance_Score']
METABOLIC_FEATURES=['RIDAGEYR','RIAGENDR','BMXWT','BMXHT','BMXBMI','DR1TKCAL','DR1TPROT','DR1TCARB','DR1TTFAT','DR1TFIBE','LBXTC','LBDHDD','LBDLDL','LBXTR']
NUTRITION=['Calories','Protein','Fat','Carbohydrates','Fiber','Sodium']
RATIOS={'Breakfast':.25,'Lunch':.35,'Snack':.10,'Dinner':.30}
ALLERGENS={'Milk':['milk','cheese','yogurt','butter','cream','whey'],'Peanut':['peanut','groundnut'],'Tree Nut':['almond','cashew','walnut','pistachio','hazelnut'],'Soy':['soy','soya','tofu'],'Egg':['egg'],'Fish':['fish','salmon','tuna','sardine'],'Shellfish':['shrimp','prawn','crab','lobster'],'Wheat':['wheat','bread','flour','pasta']}

@st.cache_resource
def load_models(): return joblib.load(DIET_MODEL_PATH), joblib.load(METABOLIC_MODEL_PATH)
@st.cache_data
def load_foods():
    df=pd.read_csv(FOOD_PATH,low_memory=False)
    for c in NUTRITION: df[c]=pd.to_numeric(df[c],errors='coerce')
    df=df[df.Calories.notna() & (df.Calories>0)]
    df=df[df.Protein.notna() & (df.Fat.notna()) & (df.Carbohydrates.notna())]
    return df.reset_index(drop=True)
def bmi(w,h): return round(w/((h/100)**2),2)
def filter_allergies(df, allergies):
    out=df.copy()
    for a in allergies:
        pat='|'.join(re.escape(x) for x in ALLERGENS[a])
        out=out[~out.description.astype(str).str.contains(pat,case=False,na=False)]
    return out.reset_index(drop=True)
def score(r,d):
    c=float(r.Carbohydrates); p=float(r.Protein); f=float(r.Fat); fi=0 if pd.isna(r.Fiber) else float(r.Fiber); s=0 if pd.isna(r.Sodium) else float(r.Sodium)
    if d=='Low_Carb': return (50 if c<=10 else 35 if c<=20 else 20 if c<=40 else 5 if c<=60 else 0)+min(p,30)+min(fi*2,15)
    if d=='Low_Sodium': return (50 if s<=140 else 35 if s<=300 else 20 if s<=500 else 5 if s<=700 else 0)+min(p,30)+min(fi*2,15)
    return min(p*1.5,30)+min(fi*2,20)+(20 if 10<=c<=60 else 0)+(15 if 5<=f<=30 else 0)+(10 if s<=500 else 0)
def meal_plan(db,diet,calories,allergies,n=3):
    safe=filter_allergies(db,allergies)
    if len(safe)<n*4: raise ValueError('Not enough foods remain after allergy filtering.')
    plan=[]; used=set()
    for meal,ratio in RATIOS.items():
        target=calories*ratio; c=safe[~safe.fdc_id.isin(used)].copy(); c['score']=c.apply(lambda r:score(r,diet),axis=1); c['dist']=(c.Calories-target/n).abs(); c['rank']=c.score-c.dist/max(target/n,1)*20; sel=c.sort_values('rank',ascending=False).head(n)
        remaining=target
        for i,(_,r) in enumerate(sel.iterrows()):
            portion=(remaining/r.Calories*100) if i==n-1 else ((target/n)/r.Calories*100)
            portion=float(np.clip(portion,20,500)); factor=portion/100; cal=float(r.Calories)*factor; remaining-=cal; used.add(r.fdc_id)
            plan.append({'Meal':meal,'Food':r.description,'Portion_g':round(portion,1),'Calories':round(cal,1),'Protein':round(r.Protein*factor,1),'Fat':round(r.Fat*factor,1),'Carbohydrates':round(r.Carbohydrates*factor,1),'Fiber':np.nan if pd.isna(r.Fiber) else round(r.Fiber*factor,1),'Sodium':np.nan if pd.isna(r.Sodium) else round(r.Sodium*factor,1)})
    return pd.DataFrame(plan)

def make_user(a,g,w,h,d,s,act,cal,chol,bp,glu,res,allerg,cuisine,ex,adh,imb):
    return {'Age':a,'Gender':g,'Weight_kg':w,'Height_cm':h,'BMI':bmi(w,h),'Disease_Type':d,'Severity':s,'Physical_Activity_Level':act,'Daily_Caloric_Intake':cal,'Cholesterol_mg/dL':chol,'Blood_Pressure_mmHg':bp,'Glucose_mg/dL':glu,'Dietary_Restrictions':res,'Allergies':', '.join(allerg) if allerg else 'None','Preferred_Cuisine':cuisine,'Weekly_Exercise_Hours':ex,'Adherence_to_Diet_Plan':adh,'Dietary_Nutrient_Imbalance_Score':imb}

try: diet_model,metabolic_model=load_models(); food_db=load_foods()
except Exception as e: st.error(f'Files/model error: {e}'); st.stop()

st.title('🥗 NutriPred'); st.caption('AI-assisted nutrition recommendation and meal-planning prototype')
with st.sidebar:
    st.header('Patient Profile'); age=st.number_input('Age',1,120,35); gender=st.selectbox('Gender',['Male','Female']); weight=st.number_input('Weight (kg)',20.,300.,75.); height=st.number_input('Height (cm)',80.,250.,175.); st.metric('BMI',f'{bmi(weight,height):.2f}')
    disease=st.selectbox('Disease Type',['None','Diabetes','Hypertension','Obesity']); severity=st.selectbox('Severity',['None','Mild','Moderate','Severe']); activity=st.selectbox('Physical Activity',['Sedentary','Moderate','Active']); calories=st.number_input('Daily Caloric Target (kcal)',800.,6000.,2200.,step=50.)
    cholesterol=st.number_input('Total Cholesterol (mg/dL)',50.,500.,190.); bp=st.number_input('Blood Pressure (mmHg)',50.,250.,120.); glucose=st.number_input('Glucose (mg/dL)',40.,500.,100.); restriction=st.selectbox('Dietary Restriction',['None','Low_Sodium','Low_Sugar']); allergies=st.multiselect('Allergies',list(ALLERGENS)); cuisine=st.selectbox('Preferred Cuisine',['Indian','Chinese','Italian','Mexican']); exercise=st.number_input('Weekly Exercise (hours)',0.,50.,4.); adherence=st.slider('Diet Plan Adherence (%)',0,100,80); imbalance=st.slider('Nutrient Imbalance Score',0,100,20); run=st.button('Generate NutriPred Plan',type='primary',use_container_width=True)
if run:
    user=make_user(age,gender,weight,height,disease,severity,activity,calories,cholesterol,bp,glucose,restriction,allergies,cuisine,exercise,adherence,imbalance); udf=pd.DataFrame([user],columns=DIET_FEATURES)
    try:
        pred=diet_model.predict(udf)[0]; probs=diet_model.predict_proba(udf)[0]; pdf=pd.DataFrame({'Diet':diet_model.classes_,'Probability (%)':probs*100}).sort_values('Probability (%)',ascending=False)
        # The original metabolic target/unit was not recovered from the available files; display as model output only.
        met={'RIDAGEYR':age,'RIAGENDR':1 if gender=='Male' else 2,'BMXWT':weight,'BMXHT':height,'BMXBMI':user['BMI'],'DR1TKCAL':calories,'DR1TPROT':max(.8*weight,1),'DR1TCARB':max(.45*calories/4,1),'DR1TTFAT':max(.28*calories/9,1),'DR1TFIBE':30.,'LBXTC':cholesterol,'LBDHDD':50.,'LBDLDL':115.,'LBXTR':125.}
        metabolic=float(metabolic_model.predict(pd.DataFrame([met],columns=METABOLIC_FEATURES))[0]); plan=meal_plan(food_db,pred,calories,allergies); totals=plan[NUTRITION].sum(skipna=True); ms=plan.groupby('Meal',as_index=False).Calories.sum(); ms['Target']=ms.Meal.map({m:calories*r for m,r in RATIOS.items()})
        st.success('NutriPred plan generated successfully.'); a,b,c,d=st.columns(4); a.metric('BMI',f'{user["BMI"]:.2f}'); b.metric('Recommended Diet',str(pred)); c.metric('Top Confidence',f'{pdf.iloc[0]["Probability (%)"]:.1f}%'); d.metric('Daily Calories',f'{totals.Calories:.1f} kcal')
        t1,t2,t3,t4=st.tabs(['Recommendation','Meal Plan','Nutrition','Model Details'])
        with t1: st.subheader('Diet Recommendation'); st.write(f'**Recommended category:** {pred}'); st.bar_chart(pdf.set_index('Diet')['Probability (%)']); st.dataframe(pdf,use_container_width=True,hide_index=True)
        with t2: st.subheader('Daily Meal Plan'); st.dataframe(plan,use_container_width=True,hide_index=True); st.subheader('Calories by Meal'); st.bar_chart(ms.set_index('Meal')[['Calories','Target']])
        with t3:
            x1,x2,x3=st.columns(3); x1.metric('Protein',f'{totals.Protein:.1f} g'); x2.metric('Carbohydrates',f'{totals.Carbohydrates:.1f} g'); x3.metric('Fat',f'{totals.Fat:.1f} g'); x4,x5,x6=st.columns(3); x4.metric('Fiber',f'{totals.Fiber:.1f} g'); x5.metric('Sodium',f'{totals.Sodium:.1f} mg'); x6.metric('Calorie Difference',f'{totals.Calories-calories:+.1f} kcal'); st.bar_chart(pd.DataFrame({'Nutrient':['Protein','Fat','Carbohydrates','Fiber'],'Grams':[totals.Protein,totals.Fat,totals.Carbohydrates,totals.Fiber]}).set_index('Nutrient'))
        with t4: st.write('**Diet model:** Random Forest classifier (200 trees)'); st.write(f'**Classes:** {list(diet_model.classes_)}'); st.write('**Metabolic model:** Random Forest regressor (200 trees)'); st.write(f'**Metabolic model output:** {metabolic:.2f}'); st.warning('The original training target/unit for the metabolic model was not recovered from the available project files. This value is therefore shown as a model output, not as a clinically validated glucose diagnosis.'); st.write(f'**Food database:** {len(food_db):,} foods available')
        st.download_button('Download Meal Plan CSV',plan.to_csv(index=False),'nutripred_meal_plan.csv','text/csv')
    except Exception as e: st.error(f'Could not generate plan: {e}')
else: st.info('Enter the patient profile in the sidebar and click Generate NutriPred Plan.')
st.divider(); st.caption('Prototype only. Not a substitute for medical or dietary advice.')
