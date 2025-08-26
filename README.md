# SME Business Helper App

A **machine learning-based forecasting solution** for small and medium shopkeepers to predict sales and manage inventory more effectively.  
The app helps shopkeepers make **data-driven decisions** by analyzing past purchase/sales data and forecasting future demand.

---

## 📌 Features
- 📊 Sales forecasting using **Linear Regression, XGBoost, and Time-Series models**  
- 🎯 Context-aware predictions (festivals, inflation, seasonal demand)  
- 📦 Inventory management suggestions  
- 📝 Simple interface with `app.py`  

---

## ⚙️ Installation

### 1. Clone the repository
```bash
git clone https://github.com/bishnu1710/SME_business_helper_app.git
cd SME_business_helper_app
```
### Step 2: Setup venv
```bash
python -m venv venv
```
# On Linux/Mac
```bash
source venv/bin/activate
```
# On Windows
```bash
venv\Scripts\activate
```
### Step 3: Install dependencies
```bash
pip install -r requirements.txt
```
### run
```bash
python app.py
````

### 📂 Project Structure
SME_business_helper_app/
│── app.py              # Main application script
│── requirements.txt    # Python dependencies
│── venv/               # Virtual environment (ignored in git)
│── .gitignore
│── README.md
### streamlit
[streamlit_app](https://smebusinessapp-bishnu.streamlit.app/)