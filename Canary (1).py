#old model and you want to deploy a new model, to migrate to the new model happens
#gradually (5%, 10%, 20%, 50%, 75%, 100%) Pilot
#monitor the new model with each deploy, if the new model is not acting properly
#then rollback (go back to the old model, fix the new model then redeploy the same 
#way again)
#Canary Release: a deployment strategy where a new version of AI model is released
#gradually to a subset of users before a full rollout. this reduces the risk and allows
#monitoring of performance before widespread adoption. 

#With Azure
#1. Create Two model (old, new)
#2. upload the models in Github each in a separate repository 
#3. use Azure to deploy the models and mointor the performance (SAIT account)

#With Grafana and Prometheus:
#1. Create two models (old, new)
#2. create a Flask API to set the predications and randomly assigns the usage 
#percentage
#3. using grafana and prometheus, monitor the performance of the program being 
#deployed 

#Create the Two Models 
import joblib
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Load dataset
iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2)

# Train old model
old_model = RandomForestClassifier(n_estimators=10)
old_model.fit(X_train, y_train)
joblib.dump(old_model, "old_model.pkl")

# Train new model (improved version)
new_model = RandomForestClassifier(n_estimators=50)  # More trees = better accuracy
new_model.fit(X_train, y_train)
joblib.dump(new_model, "new_model.pkl")

print("Old and new models saved.")


#With Azure
#1. Create Two model (old, new)
#2. upload the models in Github each in a separate repository -> create a new
#repository -> upload the file -> choose the file -> choose one of the models -> commit
#create another repository for the other model to upload there (new model, old model)
#3. use Azure to deploy the models and mointor the performance (SAIT account)
#3.1. suscribe with Azure with SAIT account 
#4. by default it will create a student azure subscription, use it for any azure 
#services
#5. app services -> create -> choose the subscription (azure student)
#-> create a new resource (use dashes) -> write an instance name (any name)
#-> select the runtime (python 3.12) -> choose the region close to you 
#-> create a new pricing plan -> select the pricing plan desired (Free plan)
#-> Monitor Tab choose yes to monitor and have insights for the model 
#-> review and create -> create 
#6. app services -> select the model created -> deployment -> deployment center
#-> choose source -> GitHub -> login to GitHub -> select organization (your account)
#-> select repository -> the created repository containing one of the models
#-> select main -> save -> it will deploy the model and make it start working 
#7. Do steps 5 and 6 for both the old and new models one for each 