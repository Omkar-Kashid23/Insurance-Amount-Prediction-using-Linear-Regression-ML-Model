# 💼 Insurance Charge Prediction — AI x Business Dashboard

A **production-ready Streamlit application** that predicts **insurance charges** using a trained Machine Learning model (`best_model_for_insurance.pkl`).  
Designed with a **hybrid AI × Business dashboard** aesthetic, this project demonstrates top 1–5% level coding clarity, structure, and deployment quality.

---

## 🚀 Features

✅ **Two Prediction Modes**
- 🧮 **Single Input** – Enter individual parameters like age, BMI, and smoker status.
- 📁 **Batch Mode** – Upload a CSV file and get instant predictions for multiple records.

✅ **Modern Dashboard Design**
- Professional **AI x Business** hybrid layout with KPI metrics.
- Intuitive sidebar controls and responsive UI.
- Compact KPI cards (Predicted charge, Median, Std. Dev).

✅ **Smart Model Handling**
- Auto-detects your model’s expected features from scikit-learn Pipelines.
- Caches the model efficiently using Streamlit’s resource caching.

✅ **Explainability**
- Lightweight rule-based “Why this prediction?” insights for transparency.
- Human-readable interpretations without heavy dependencies (no SHAP required).

✅ **Production-Ready**
- Fully dockerized.
- Clean `requirements.txt`.
- Works seamlessly with Streamlit Cloud, GitHub Actions, or any Docker-based deploy.

---

## 🧠 Model Information

This app uses a trained ML model (`best_model_for_insurance.pkl`) that predicts **insurance charges** based on:
- `age` — numerical  
- `bmi` — numerical  
- `children` — numerical  
- `sex` — categorical  
- `smoker` — categorical (`yes`/`no`)  
- `region` — categorical (`southwest`, `southeast`, `northwest`, `northeast`)

> The model file should be placed in the same directory as `app.py`.

---

## 🛠️ Installation

### 1️⃣ Clone the repository
```bash
git clone https://github.com/yourusername/insurance-charge-dashboard.git
cd insurance-charge-dashboard

### 2️⃣ Add your trained model

Place your trained model file as:

```
best_model_for_insurance.pkl
```

### 3️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

### 4️⃣ Run the app

```bash
streamlit run app.py
```

Then open your browser at **[http://localhost:8501](http://localhost:8501)**

---

## 🐳 Docker Deployment

You can run the app in a containerized environment for production-grade deployment.

### Build the image

```bash
docker build -t insurance-app:latest .
```

### Run the container

```bash
docker run -p 8501:8501 insurance-app:latest
```

### Access

Navigate to 👉 [http://localhost:8501](http://localhost:8501)

---

## 📊 Example Inputs

| age | bmi  | children | sex    | smoker | region    |
| --- | ---- | -------- | ------ | ------ | --------- |
| 22  | 19.8 | 0        | male   | no     | northwest |
| 57  | 31.2 | 2        | female | yes    | southeast |

---

## 🧾 Output Example

| age | bmi  | children | sex    | smoker | region    | predicted_charge |
| --- | ---- | -------- | ------ | ------ | --------- | ---------------- |
| 22  | 19.8 | 0        | male   | no     | northwest | ₹9,870           |
| 57  | 31.2 | 2        | female | yes    | southeast | ₹46,350          |

---

## 🧩 Project Structure

```
📂 Insurance-Charge-Prediction
 ┣ 📄 app.py
 ┣ 📄 best_model_for_insurance.pkl
 ┣ 📄 requirements.txt
 ┣ 📄 Dockerfile
 ┣ 📄 README.md
 ┗ 📄 Insurance project.ipynb

```

---

## ⚙️ Environment Variables

| Variable     | Default                        | Description                      |
| ------------ | ------------------------------ | -------------------------------- |
| `MODEL_PATH` | `best_model_for_insurance.pkl` | Path to your serialized ML model |

---

## 🧠 Tech Stack

* **Frontend** – Streamlit (Modern UI + KPI dashboards)
* **Backend** – Python 3.10, scikit-learn
* **Containerization** – Docker
* **Data** – CSV/Manual input forms
* **Deployment** – Streamlit Cloud / Docker / GitHub Actions

---

## 🧑‍💻 Author

**Omkar Kashid**
📧 *[okashid104@gmail.com](mailto:okashid104@gmail.com)*
🚀 Passionate about **Machine Learning**, **AI**, and **clean production-grade engineering**.

---

## 🪄 Future Enhancements

* Add authentication (Streamlit + OAuth)
* Integrate SHAP/ELI5 explainers for deeper insights
* Add Prometheus/Grafana metrics for MLOps monitoring
* CI/CD via GitHub Actions
* Cloud storage integration for batch inputs

---

## 🧾 License

This project is licensed under the **MIT License** – feel free to use, modify, and share.

---

> ✨ *“Simple, explainable AI beats complex black boxes.”*
> — *Omkar Kashid*
