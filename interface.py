def get_patient_data():
    print("Enter patient details:")
    Pregnancies = int(input("Pregnancies: "))
    Glucose = float(input("Glucose: "))
    BloodPressure = float(input("BloodPressure: "))
    SkinThickness = float(input("SkinThickness: "))
    Insulin = float(input("Insulin: "))
    BMI = float(input("BMI: "))
    DiabetesPedigreeFunction = float(input("DiabetesPedigreeFunction: "))
    Age = int(input("Age: "))
    
    patient_data = [[Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]]
    patient_data = scaler.transform(patient_data)
    return patient_data

def predict_disease():
    patient_data = get_patient_data()
    prediction = model.predict(patient_data)
    if prediction[0] == 1:
        print("The patient is likely to have diabetes.")
    else:
        print("The patient is unlikely to have diabetes.")

# Run the expert system
predict_disease()
