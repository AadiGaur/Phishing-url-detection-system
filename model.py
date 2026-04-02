import pandas as pd                     #read and handle dataset
from sklearn.model_selection import train_test_split    #divide data into training and testing 
from sklearn.ensemble import RandomForestClassifier     #ML algorithm 
import pickle   #save the trained model 


df = pd.read_csv("Phishing_Legitimate_full.csv")  #loads the dataset 


df = df.drop("Index", axis=1) # remove unnecesarry columns 


X = df.iloc[:, :-1]   #//features column (url characterstics check)
y = df.iloc[:, -1]    #//Labels (safe or phishing)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)   #//Data set (80% training) and (20% testing)


model = RandomForestClassifier()  #create Random forest algorithm 
model.fit(X_train, y_train) #learn patterns from the training


accuracy = model.score(X_test, y_test)
print("Model Accuracy:", accuracy)

pickle.dump(model, open("phishing_model.pkl", "wb"))

print("Model saved successfully")