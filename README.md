# Student Performance Prediction

## Problem description
Build a regression model to predict student exam scores from demographic and study-related factors. The project covers data preparation, exploratory analysis, feature importance, and model selection/tuning.

## Project contents
- `notebook.ipynb`: Data preparation/cleaning, EDA, feature importance analysis, and model selection/tuning.
- `data/StudentPerformanceFactors.csv`: Dataset used in the notebook (committed to this repo).
- `train.py`: Trains the final model and saves a model artifact.
- `predict.py`: Loads the model and serves predictions via a web service.
- `requirements.txt`: Python dependencies.
- `Dockerfile`: Container image for the prediction service.
- `model.pkl`: Generated model artifact (created by running `train.py`).

## How to run
1. Create and activate a Python environment (venv/conda/etc.).
2. Install dependencies:
   - `pip install -r requirements.txt`
3. Launch Jupyter and open the notebook:
   - `jupyter notebook`
   - open `notebook.ipynb`

## Data
The dataset is committed at `data/StudentPerformanceFactors.csv`. The notebook and training script load it from:
`data/StudentPerformanceFactors.csv`.

## Training the final model
Run the training script to create `model.pkl`:
```
python train.py --data-path data/StudentPerformanceFactors.csv --model-path model.pkl
```

## Prediction service
Start the service (expects `model.pkl` in the project root):
```
python predict.py --model-path model.pkl
```

Example request:
```
curl -X POST http://localhost:9696/predict   -H "Content-Type: application/json"   -d '{
    "record": {
      "Hours_Studied": 10,
      "Attendance": 90,
      "Parental_Involvement": "Medium",
      "Access_to_Resources": "High",
      "Extracurricular_Activities": "No",
      "Sleep_Hours": 7,
      "Previous_Scores": 80,
      "Motivation_Level": "High",
      "Internet_Access": "Yes",
      "Tutoring_Sessions": 1,
      "Family_Income": "Medium",
      "Teacher_Quality": "High",
      "School_Type": "Public",
      "Peer_Influence": "Positive",
      "Physical_Activity": 3,
      "Learning_Disabilities": "No",
      "Parental_Education_Level": "College",
      "Distance_from_Home": "Near",
      "Gender": "Female"
    }
  }'
```

## Docker
Build after generating `model.pkl`:
```
docker build -t student-performance .
```

Run the service:
```
docker run -p 9696:9696 student-performance
```

## Deployment
Add one of the following:
- Deployment URL to a running service, or
- Video or image showing interaction with the deployed service.

### Railway (live testing)
Public URL: `https://mlzcampcapstone2-production.up.railway.app`

Health:
```
curl https://mlzcampcapstone2-production.up.railway.app/health
```

Predict:
```
curl -X POST https://mlzcampcapstone2-production.up.railway.app/predict \
  -H "Content-Type: application/json" \
  -d '{
    "record": {
      "Hours_Studied": 10,
      "Attendance": 90,
      "Parental_Involvement": "Medium",
      "Access_to_Resources": "High",
      "Extracurricular_Activities": "No",
      "Sleep_Hours": 7,
      "Previous_Scores": 80,
      "Motivation_Level": "High",
      "Internet_Access": "Yes",
      "Tutoring_Sessions": 1,
      "Family_Income": "Medium",
      "Teacher_Quality": "High",
      "School_Type": "Public",
      "Peer_Influence": "Positive",
      "Physical_Activity": 3,
      "Learning_Disabilities": "No",
      "Parental_Education_Level": "College",
      "Distance_from_Home": "Near",
      "Gender": "Female"
    }
  }'
```

PowerShell (recommended on Windows):
```
$body = @{
  record = @{
    Hours_Studied = 10
    Attendance = 90
    Parental_Involvement = "Medium"
    Access_to_Resources = "High"
    Extracurricular_Activities = "No"
    Sleep_Hours = 7
    Previous_Scores = 80
    Motivation_Level = "High"
    Internet_Access = "Yes"
    Tutoring_Sessions = 1
    Family_Income = "Medium"
    Teacher_Quality = "High"
    School_Type = "Public"
    Peer_Influence = "Positive"
    Physical_Activity = 3
    Learning_Disabilities = "No"
    Parental_Education_Level = "College"
    Distance_from_Home = "Near"
    Gender = "Female"
  }
} | ConvertTo-Json -Depth 5

Invoke-RestMethod -Uri https://mlzcampcapstone2-production.up.railway.app/predict -Method Post -ContentType "application/json" -Body $body
```
