from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
import torch
from MnistModule import MnistModule
if __name__ == '__main__':
    mnistModule = MnistModule()
    model = torch.hub.load('/content/drive/MyDrive/digit_classifier_model.pt')
    #test path
    test_images_path = '/content/drive/MyDrive/t10k-images-idx3-ubyte.gz'
    test_labels_path = '/content/drive/MyDrive/t10k-labels-idx1-ubyte.gz'
    test_images, test_labels = mnistModule.get_images_labels(test_images_path, test_labels_path, 10000)

    predictions = mnistModule.predict(test_images)

    mnistModule.get_accuracy(test_labels)

    mnistModule.get_f1_score(test_labels)

    mnistModule.get_precision_score(test_labels)
