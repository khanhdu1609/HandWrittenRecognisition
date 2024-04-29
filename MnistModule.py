import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torchvision import transforms
from torch import nn,save,load
import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch import nn
from PIL import Image
class MnistModule:
  def __init__(self, num_epochs=100, batch_size=64):
    self.num_epochs = num_epochs
    self.batch_size = batch_size
    self.inm_transformer = transforms.Compose([transforms.ToTensor()])
    self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  def get_mnist_data(self, images_path, labels_path, num_images, shuffle=False, _is=True, image_size=28):
      # read data
      import gzip # to decompress gz (zip) file
      # open file training to read training data
      f_images = gzip.open(images_path,'r')
      # skip 16 first bytes because these are not data, only header infor
      f_images.read(16)
      # general: read num_images data samples if this parameter is set;
      # if not, read all (60000 training or 10000 test)
      real_num = num_images if not shuffle else (60000 if _is else 10000)
      # read all data to buf_images (28x28xreal_num)
      buf_images = f_images.read(image_size * image_size * real_num)
      # images
      images = np.frombuffer(buf_images, dtype=np.uint8).astype(np.float32)
      images = images.reshape(real_num, image_size, image_size,)
      # Read labels
      f_labels = gzip.open(labels_path,'r')
      f_labels.read(8)
      labels = np.zeros((real_num)).astype(np.int64)
      # rearrange to correspond the images and labels
      for i in range(0, real_num):
        buf_labels = f_labels.read(1)
        labels[i] = np.frombuffer(buf_labels, dtype=np.uint8).astype(np.int64)

      # shuffle to get random images data
      if shuffle is True:
        rand_id = np.random.randint(real_num, size=num_images)
        images = images[rand_id, :]
        labels = labels[rand_id,]
      images /= 255
      return images, labels
  def get_images_labels(self, images_path, labels_path, num_imgs):
    images, labels = self.get_mnist_data(images_path, labels_path, num_imgs, shuffle=False)
    return images, labels
  def load_data(self, images, labels):
    
    #Add image tensor to dataset
    dataset = ()
    for i in range(60000):
      dataset += ((self.img_transformer(images[i]), int(labels[i])),)
    data_loader = DataLoader(dataset, batch_size=self.batch_size, shuffle = True)
    return data_loader
  def train(self, model, data_loader):
    model = model.to(self.device)
    optimizer = Adam(model.parameters(), lr=0.001)
    loss_fn = nn.CrossEntropyLoss()
    for epoch in range(self.num_epochs):
      training_acc = 0
      for images, labels in data_loader:
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        images, labels = images.to(self.device), labels.to(self.device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = loss_fn(outputs, labels)
        total_loss += loss.item()
        loss.backward()
        optimizer.step()
        # Compute accuracy
        _, predicted = torch.max(outputs, 1)
        total_correct += (predicted == labels).sum().item()
        total_samples += labels.size(0)

        # Calculate average loss and accuracy
      average_loss = total_loss / len(data_loader)
      accuracy = total_correct / total_samples * 100  # Percentage
      print(f"Epoch {epoch + 1}/{self.num_epochs}, Loss: {average_loss:.4f}, Accuracy: {accuracy:.2f}%")
    print("Training complete")
    return model
  def save(self, model, path):
    torch.save(model.state_dict(), path)
    print(f"Saved to {path}")
  def predict(self, model, test_images):
    predictions = np.zeros(10000)
    for i in range(10000):
      img = Image.fromarray(test_images[i])
      img_tensor = self.img_transformer(img).unsqueeze(0).to(self.device)
      output = model(img_tensor)
      predicted_label = torch.argmax(output)
      predictions[i] = predicted_label
    return predictions
  def get_accuracy(self, predictions, test_labels):
    return accuracy_score(predictions, test_labels)
  def get_f1_score(self, predictions, test_labels):
    return f1_score(predictions, test_labels, average='macro')
  def get_recall_score(self, predictions, test_labels):
    return recall_score(predictions, test_labels, average='macro')
  def get_precision_score(self, predictions, test_labels):
    return precision_score(predictions, test_labels, average='macro')
