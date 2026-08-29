from google.colab import drive, files
import os, shutil, zipfile

drive.mount('/content/drive')
BASE='/content/drive/MyDrive/nutripred_dtaset'; APP='/content/nutripred_app'; os.makedirs(APP,exist_ok=True)
# Upload/copy app.py and requirements.txt to /content before running this cell.
for name in ['app.py','requirements.txt']:
    src='/content/'+name
    if os.path.exists(src): shutil.copy2(src,APP+'/'+name)
    elif not os.path.exists(APP+'/'+name): raise FileNotFoundError(src+' not found')
files_to_copy={'trained_models/diet_recommendation_model.pkl':'diet_recommendation_model.pkl','trained_models/metabolic_glucose_model.pkl':'metabolic_glucose_model.pkl','processed_food_nutrition.csv':'processed_food_nutrition.csv'}
for rel,name in files_to_copy.items():
    src=os.path.join(BASE,rel); dst=os.path.join(APP,name)
    if not os.path.exists(src): raise FileNotFoundError(src)
    shutil.copy2(src,dst)
zip_path='/content/nutripred_streamlit.zip'
with zipfile.ZipFile(zip_path,'w',zipfile.ZIP_DEFLATED) as z:
    for f in os.listdir(APP):
        p=os.path.join(APP,f)
        if os.path.isfile(p): z.write(p,arcname=f)
print('Created:',zip_path); files.download(zip_path)
