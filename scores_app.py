import pandas as pd
import streamlit as st 
import numpy as np
from feature_engine.outliers import Winsorizer

from sqlalchemy import create_engine
import pickle, joblib

impute = joblib.load('meanimpute')
winsor = joblib.load('winzor')
poly_model = pickle.load(open('poly_model.pkl', 'rb'))


def predict_Scores(data,user,pw,db):

    engine = create_engine(f"mysql+pymysql://{user}:{pw}@localhost/{db}")
                    
    clean1 = pd.DataFrame(impute.transform(data), columns = data.select_dtypes(exclude = ['object']).columns)   
    clean2 = pd.DataFrame(winsor.transform(clean1), columns = clean1.columns)    
    prediction = pd.DataFrame(poly_model.predict(clean2), columns = ['Pred_Scores'])
    
    final = pd.concat([prediction, data], axis = 1)
    final.to_sql('Scores_predictons', con = engine, if_exists = 'replace', chunksize = 1000, index= False)

    return final



def main():
    st.title("Scores prediction")
    st.sidebar.title("Scores Efficiency prediction")
    
    html_temp = """
    <div style="background-color:tomato;padding:10px">
    <h2 style="color:white;text-align:center;">Scores prediction ML App </h2>
    </div>
    """
    st.markdown(html_temp,unsafe_allow_html = True)
    uploadedFile = st.sidebar.file_uploader("Choose a file", type=['csv','xlsx'], accept_multiple_files=False, key="fileUploader")
    if uploadedFile is not None :
        try:

            data = pd.read_csv(uploadedFile)
        except:
                try:
                    data = pd.read_excel(uploadedFile)
                except:      
                    data = pd.DataFrame()
        
        
    else:
        st.sidebar.warning("You need to upload a CSV or an Excel file.")
    
    html_temp = """
    <div style="background-color:tomato;padding:10px">
    <p style="color:white;text-align:center;">Add DataBase Credientials </p>
    </div>
    """
    st.sidebar.markdown(html_temp, unsafe_allow_html = True)
            
    user = st.sidebar.text_input("user", "Type Here")
    pw = st.sidebar.text_input("password", "Type Here")
    db = st.sidebar.text_input("database", "Type Here")
    
    result = ""
    
    if st.button("Predict"):
        result = predict_Scores(data, user, pw, db)
        #st.dataframe(result) or
        #st.table(result.style.set_properties(**{'background-color': 'white','color': 'black'}))
                           
        import seaborn as sns
        cm = sns.light_palette("red", as_cmap = True)
        
        # Set precision for float values
        pd.options.display.float_format = '{:.2f}'.format
        
        st.table(result.style.background_gradient(cmap=cm))

if __name__=='__main__':
    main()


