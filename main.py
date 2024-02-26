import os
import numpy as np
from skimage.io import imread
from skimage.transform import resize
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score,confusion_matrix




# Data preprocessing
input_dir = "/home/user/PycharmProjects/Image-Classification/animals/animals/"
input_txt = "name of the animals.txt"
labels = []
images = []
with open(input_txt, 'r') as f:
    for value in f:
        value = value.strip('\n')
        labels.append(value)

print(labels)

for category in labels:
    category_dir = os.path.join(input_dir, category)
    for file_name in os.listdir(category_dir):
        file_path = os.path.join(category_dir, file_name)
        img = imread(file_path)
        img = resize(img, (15, 15))
        images.append(img)




# Label Encoding
le = LabelEncoder()
labels_encoded = le.fit_transform(labels)

# Convert lists to numpy arrays
images = np.array(images)
labels_encoded = np.array(labels_encoded)

# Split Data
X_train, X_test, y_train, y_test = train_test_split(images, labels_encoded, test_size=0.33, random_state=0)

# Model
classifier = SVC(probability=True)
param_grid = {'C': [0.1, 1, 10, 100], 'gamma': [0.0001, 0.001, 0.1, 1], 'kernel': ['rbf', 'poly']}
model = GridSearchCV(classifier, param_grid)
model.fit(X_train,y_train)


# Prediction
y_pred = model.predict(X_test)

# Accuracy Score
acc = accuracy_score(y_test,y_pred)
print(acc)


# Confusion Matrix

matrix = confusion_matrix(y_test, y_pred)
print(matrix)