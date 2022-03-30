from unicodedata import name
import streamlit as st
import pickle


pickle_in = open("decision.pkl","rb")
classifier=pickle.load(pickle_in)
pickle_in.close()
#message
#abhinav    
pickle_pca= open("finalized_pca_model.sav","rb")
classifier_pca=pickle.load(pickle_pca)
pickle_pca.close()

pickle_ss= open("finalized_SS_model.sav","rb")
classifier_ss=pickle.load(pickle_ss) 
pickle_ss.close()


def predictdiabetes(nooftimespregnant,glucoselevel,bloodpressure,skinthickness,insulin,bmi,diabetespedigreefunction,age):
    data=[[nooftimespregnant,glucoselevel,bloodpressure,skinthickness,insulin,bmi,diabetespedigreefunction,age]]
    #print(data)
    transform_data=classifier_ss.transform(data)
    # print("value of scaler model is {}",format(transform_data))
    pca_data=classifier_pca.transform(transform_data)
    # print("value of pca model is {}",format(pca_data))
    prediction=classifier.predict(pca_data)
    # print ('-------------')
    # print (classifier.predict([[-0.66211028, -1.25212298,  0.7700672 , -0.82632298,  0.67155817,
    #    -0.91282061, -0.37403062, -0.83412818]]))
    # print("Value of decion tree model is {}",format(prediction))
    return prediction


if __name__=='__main__':
    st.title("DIABETES PREDICTION MODEL")
    st.write("Enter the details of the patient diagnostic measures: ")
    nooftimespregnant=st.number_input("No. of Times Pregnant")
    glucoselevel=st.number_input("Gluose Level")
    bloodpressure=st.number_input("Blood Pressure")
    skinthickness=st.number_input("Skin Thickness")
    insulin=st.number_input("Insulin")
    bmi=st.number_input("BMI")
    diabetespedigreefunction=st.number_input("Diabetes Pedigree Function")
    age=st.number_input("Age")
    if st.button("Predict"):
        result=predictdiabetes(nooftimespregnant,glucoselevel,bloodpressure,skinthickness,insulin,bmi,diabetespedigreefunction,age)
        if result==1:
            st.success("You are diabetic")
        if result==0:
            st.success("You are not diabetic")