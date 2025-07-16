from flask import Flask, request, redirect, url_for, send_file, flash, render_template_string
import pandas as pd
import joblib
import io

app = Flask(__name__)
app.secret_key = 'your_secret_key'

# Specify your serialized model pipeline filename here (it should include preprocessing)
MODEL_FILENAME = 'Customer Churn - EDA.ipynb'

# Load your trained model pipeline
try:
    model = joblib.load('Customer Churn - EDA.ipynb')
except FileNotFoundError:
    raise FileNotFoundError(f"Model file '{MODEL_FILENAME}' not found. Please serialize your full pipeline (including encoders) to this path.")

# Inline CSS for both pages
global_css = """
<style>
  body { font-family: Arial, sans-serif; background-color: #f4f4f4; margin:0; padding:0; }
  .container { max-width: 600px; margin:50px auto; background:#fff; padding:20px; border-radius:8px; box-shadow:0 2px 5px rgba(0,0,0,0.1); }
  h1 { text-align:center; color:#333; }
  form { display:flex; flex-direction:column; }
  label { margin-top:10px; color:#555; }
  input, select { padding:8px; margin-top:5px; border:1px solid #ddd; border-radius:4px; }
  button { margin-top:20px; padding:10px; background-color:#28a745; color:#fff; border:none; border-radius:4px; cursor:pointer; }
  button:hover { background-color:#218838; }
  .result { margin-top:20px; padding:15px; background-color:#e9ecef; border-radius:4px; }
  .errors { color:#d9534f; list-style:none; padding:0; }
  a { color:#007bff; text-decoration:none; margin-top:10px; }
  a:hover { text-decoration:underline; }
</style>
"""

# Feature list (all columns except target)
fields = [
    'gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure',
    'PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity',
    'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV',
    'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod',
    'MonthlyCharges', 'TotalCharges'
]

# HTML template strings
index_html = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Churn Prediction</title>
  {{ css }}
</head>
<body>
  <div class="container">
    <h1>Telecom Customer Churn Prediction</h1>
    <form method="post">
      {% for field in fields %}
        <label>{{ field }}:</label>
        {% if field in ['gender','Partner','Dependents','PhoneService','MultipleLines','InternetService','OnlineSecurity','OnlineBackup','DeviceProtection','TechSupport','StreamingTV','StreamingMovies','Contract','PaperlessBilling','PaymentMethod'] %}
          <input type="text" name="{{ field }}" value="{{ values.get(field, '') }}" required>
        {% else %}
          <input type="number" step="any" name="{{ field }}" value="{{ values.get(field, '') }}" required>
        {% endif %}
      {% endfor %}
      <button type="submit">Predict</button>
    </form>
    {% if result is not none %}
      <div class="result">
        <p><strong>Prediction:</strong> {{ result }}</p>
        <p><strong>Probability:</strong> {{ probability }}</p>
      </div>
    {% endif %}
    <a href="{{ url_for('batch') }}">Batch Prediction (CSV)</a>
  </div>
</body>
</html>
"""

batch_html = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Batch Churn Prediction</title>
  {{ css }}
</head>
<body>
  <div class="container">
    <h1>Batch Churn Prediction</h1>
    <form method="post" enctype="multipart/form-data">
      <input type="file" name="file" accept=".csv" required>
      <button type="submit">Upload & Predict</button>
    </form>
    {% with messages = get_flashed_messages() %}
      {% if messages %}
        <ul class="errors">
          {% for msg in messages %}<li>{{ msg }}</li>{% endfor %}
        </ul>
      {% endif %}
    {% endwith %}
    <a href="{{ url_for('index') }}">Single Prediction</a>
  </div>
</body>
</html>
"""

@app.route('/', methods=['GET','POST'])
def index():
    result = None
    probability = None
    values = {}
    if request.method == 'POST':
        # collect raw inputs
        values = {f: request.form.get(f) for f in fields}
        df = pd.DataFrame([values])
        # let your pipeline handle preprocessing
        pred = model.predict(df)[0]
        proba = model.predict_proba(df)[0,1]
        result = 'Churn' if pred==1 else 'No Churn'
        probability = f"{proba:.2f}"
    return render_template_string(index_html, css=global_css, fields=fields, result=result, probability=probability, values=values)

@app.route('/batch', methods=['GET','POST'])
def batch():
    if request.method == 'POST':
        file = request.files.get('file')
        if not file or file.filename=='':
            flash('No file selected')
            return redirect(request.url)
        df = pd.read_csv(file)
        preds = model.predict(df)
        probs = model.predict_proba(df)[:,1]
        df['Prediction'] = preds
        df['Probability'] = probs
        buf = io.StringIO()
        df.to_csv(buf,index=False)
        buf.seek(0)
        return send_file(io.BytesIO(buf.getvalue().encode()), mimetype='text/csv', as_attachment=True, download_name='batch_predictions.csv')
    return render_template_string(batch_html, css=global_css)

if __name__ == '__main__':
    app.run(debug=True)
