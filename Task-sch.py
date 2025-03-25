#This code will retrain the AI model and save the updated version periodically 
#Schedule the script to run automitcally as specific intervals 

#Using Windows:
#1. open the run app (win+r) or search for it
#2. type taskschd.msc in the run app --> enter
#3. create a new basic task -> name the task with any name -> add a description (optional)
#4. Set the occurance -> daily, weekly,..etc. 
#5. Set time and date -> when do you want to start the operation -> syncronize (optional)
#6. Set the Action -> start a program 
#7. Browse for python program to run 
#6. arguments are optional 
#7. finish
#8. verify the task -> open the task scheduler and look under the active tasks

#Using Linux -> Create a Cron Job

import joblib
import datetime
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris

# Load dataset
iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2)

# Train model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Save updated model with timestamp
#extract date and time from the system
timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
#set the file name with the timestamp
model_filename = f'updated_model_{timestamp}.pkl'
#save the model
joblib.dump(model, model_filename)

print(f"Model retrained and saved as {model_filename}")