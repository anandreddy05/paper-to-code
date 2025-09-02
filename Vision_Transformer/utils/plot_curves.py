from typing import Dict,List
import matplotlib.pyplot as plt

def plot_predictions(results:Dict[str,List[float]]):
  """ Plots training curves of a results dictionary """
  loss = results["train_loss"]
  test_loss = results["test_loss"]

  acc = results["train_acc"]
  test_acc = results["test_acc"]

  epochs = range(len("train_loss"))

  plt.figure(figsize=(15,4))
  plt.subplot(1,2,1)
  plt.plot(epochs,loss,label="train_loss")
  plt.plot(epochs,test_loss,label="test_loss")
  plt.title("Loss")
  plt.xlabel("Epochs")
  plt.legend()

  plt.subplot(1,2,2)
  plt.plot(epochs,acc,label="train_accuracy")
  plt.plot(epochs,test_acc,label="test accuracy")
  plt.title("Accuracy")
  plt.xlabel("Epochs")
  plt.legend()