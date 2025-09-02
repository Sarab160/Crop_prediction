import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

# Title
st.title("🌱 Crop Recommendation with KNN")

# Load dataset directly
df = pd.read_csv("crop.csv")

st.subheader("📊 Dataset Preview")
st.dataframe(df.head(5))

st.write(f"**Unique districts:** {df['district'].nunique()}")

# Features & Labels
x = df[["N","P","K","temperature","humidity","ph","rainfall"]]
y = df["label"]

# Encode labels
le_label = LabelEncoder()
y = le_label.fit_transform(y)

# Encode district
le_district = LabelEncoder()
district_encoded = le_district.fit_transform(df["district"])
encode_data = pd.DataFrame(district_encoded, columns=["district"])

# Combine features
X = pd.concat([x, encode_data], axis=1)
ss = StandardScaler()
X_final = ss.fit_transform(X)

# Train-test split
x_train, x_test, y_train, y_test = train_test_split(X_final, y, test_size=0.2, random_state=60)

# Sidebar parameters
st.sidebar.header("⚙️ Model Parameters")
k = st.sidebar.slider("Number of Neighbors (k)", min_value=1, max_value=50, value=20)

# Train model
knc = KNeighborsClassifier(n_neighbors=k)
knc.fit(x_train, y_train)

# Scores
st.subheader("📈 Model Performance")
st.write("**Test Score:**", knc.score(x_test, y_test))
st.write("**Train Score:**", knc.score(x_train, y_train))
st.write("**Precision:**", precision_score(y_test, knc.predict(x_test), average="macro"))
st.write("**F1 Score:**", f1_score(y_test, knc.predict(x_test), average="macro"))
st.write("**Recall:**", recall_score(y_test, knc.predict(x_test), average="macro"))

# Confusion Matrix
st.subheader("🔎 Confusion Matrix")
confusion = confusion_matrix(y_test, knc.predict(x_test))
fig, ax = plt.subplots()
sns.heatmap(confusion, annot=True, fmt="d", cmap="Blues", ax=ax)
st.pyplot(fig)

# Prediction Section
st.subheader("🌾 Try Your Own Prediction")
N = st.number_input("Nitrogen (N)", value=50)
P = st.number_input("Phosphorus (P)", value=50)
K = st.number_input("Potassium (K)", value=50)
temperature = st.number_input("Temperature (°C)", value=25.0)
humidity = st.number_input("Humidity (%)", value=50.0)
ph = st.number_input("Soil pH", value=6.5)
rainfall = st.number_input("Rainfall (mm)", value=100.0)
district = st.selectbox("District", df["district"].unique())

if st.button("Predict Crop"):
    # Encode district input
    district_val = le_district.transform([district])[0]
    input_data = [[N, P, K, temperature, humidity, ph, rainfall, district_val]]
    input_data = ss.transform(input_data)
    prediction = knc.predict(input_data)
    crop = le_label.inverse_transform(prediction)
    st.success(f"🌾 Recommended Crop: **{crop[0]}**")
