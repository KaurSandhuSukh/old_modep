#Step 2: Deploy API with Canary release strategy

#ceate a flask API that routes 5% of requrest to the new model

from flask import Flask, request, jsonify
import joblib
import numpy as np
import random

app = Flask(__name__)

# Load both models
old_model = joblib.load("old_model.pkl")
new_model = joblib.load("new_model.pkl")

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json['features']
    data = np.array(data).reshape(1, -1)

    # Canary release: 5% chance of using new model
    if random.random() < 0.05:  
        model_version = "new_model"
        prediction = new_model.predict(data)[0]
    else:
        model_version = "old_model"
        prediction = old_model.predict(data)[0]

    return jsonify({"model_version": model_version, "prediction": int(prediction)})

if __name__ == "__main__":
    app.run(debug=True)

#Step 3: gradually increase traffic to the new model
#modify the random.random() < 0.05 logic to increase
#rollout over time. 
#Example:
#Week 1: if random.random() < 0.05: (5% traffic)
#Week 2: if random.random() < 0.20: (20% traffic)
#Week 3: if random.random() < 0.50: (50% traffic)
#Week 4: Switch all traffic to new_model. use just the two statements in the if without if else

#random.random: functino to select random users to deploy the new model for
#for a complete rollout no need for the if else just the two statements inside the if
#to rollback: if False: the new model else: the old model 
#this will force the false output for the if condition which will direct the statement
#directly to the else part 
