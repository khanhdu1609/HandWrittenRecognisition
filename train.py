from model import ClassifierModel
from MnistModule import MnistModule
if __name__ == '__main__':
    # train path
    mnistModule = MnistModule()
    train_images_path = '/content/drive/MyDrive/train-images-idx3-ubyte.gz'
    train_labels_path = '/content/drive/MyDrive/train-labels-idx1-ubyte.gz'
    train_images, train_labels = mnistModule.get_images_labels(train_images_path, train_labels_path, 60000)
    train_loader = mnistModule.load_data(train_images, train_labels)
    #define model
    model = ClassifierModel()
    #train model
    model = mnistModule.train(model, train_loader)
    #save model 
    path = '/content/drive/MyDrive/digit_classifier_model.pt'
    MnistModule.save(model, path)
