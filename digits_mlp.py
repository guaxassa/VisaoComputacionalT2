import cv2
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, accuracy_score


def extract_hog_features(images):
    hog = cv2.HOGDescriptor()
    features = []
    for img in images:
        img_resized = cv2.resize(img, (64, 64))
        h = hog.compute(img_resized)
        features.append(h.flatten())
    return np.array(features)


def main():
    digits = load_digits()
    X = digits.images
    y = digits.target

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    X_train_feat = extract_hog_features(X_train)
    X_test_feat = extract_hog_features(X_test)

    clf = MLPClassifier(hidden_layer_sizes=(100,), max_iter=300, random_state=42)
    clf.fit(X_train_feat, y_train)

    y_pred = clf.predict(X_test_feat)
    print('Accuracy:', accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))


if __name__ == '__main__':
    main()
