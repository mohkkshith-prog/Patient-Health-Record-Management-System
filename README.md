# Patient-Health-Record-Management-System

=========================================

"A simple and smart way to manage patient health data digitally."

---

##  Overview

The *Patient Health Record Management System* is a Python-based desktop application developed as part of the *First Semester IDT Mini Project*.  
The system helps in digitally storing, managing, and analyzing patient health records with a user-friendly interface.

In addition to record management, the project demonstrates the use of *basic Machine Learning concepts* to predict patient health risk levels based on medical parameters.

---

## Key Features

- Add, view, update, and delete patient records
-  Search patients using ID or name
-  Local data storage using JSON files
-  Health risk prediction using Machine Learning
- Risk classification: *Low / Medium / High*
- Medical image upload (X-ray / scan demo)
-  Simple and interactive GUI using Tkinter

---

## Machine Learning Component

- Algorithm Used: *Random Forest Classifier*
- Dataset: *Synthetic health dataset (auto-generated)*
- Features considered:
  - Age
  - Body Mass Index (BMI)
  - Blood Pressure
  - Glucose Level
  - Smoking Status
  - Diabetes Indicator

The model predicts the *health risk level* of a patient based on the input parameters.

---

##  Technologies Used

- *Python 3*
- *Tkinter* – GUI
- *NumPy*
- *Scikit-learn*
- *Joblib*
- *Pillow (PIL)*
- *OpenCV*
- *JSON* – Data storage

---

##  Project Structure
Patient-Health-Record-Management-System/ │ ├── patient_healthrecord_management_system.py   # Main application ├── patients.json                               # Patient records (auto-created) ├── model.pkl                                  # Trained ML model ├── health_data.csv                            # Synthetic dataset ├── README.md └── .gitignore
Copy code

---

## How to Run the Project

### Clone the Repository
git clone https://github.com/your-username/patient-health-record-management-system.git� cd patient-health-record-management-system
Copy code

###  Install Required Libraries
pip install numpy scikit-learn pillow opencv-python joblib
Copy Coder 

### Run the Application
python patient_healthrecord_management_system.py
Copy code

---

## Usage Guide

1. Launch the application.
2. Enter patient details in the form.
3. Save records to the local database.
4. Use the *Health Risk Prediction* option to analyze patient data.
5. View predicted risk level on the interface.

---

## Academic Information

- *Course:* IDT Laboratory  
- *Semester:* First Semester  
- *Project Type:* Mini Project  
- *Purpose:* Academic and learning-oriented

---

## Author

**Your name *  
First Semester – IDT Lab  

---

##  License

This project is created for *academic purposes*.  
Free to use and modify for learning.
