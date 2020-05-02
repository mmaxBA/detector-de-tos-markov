# Emanuel Estrada
# Maximo Bautista
# Sebastian Cedeño
# Georgina González
# Francisco Márquez

from python_speech_features import mfcc, logfbank
from scipy.io import wavfile
import numpy as np
import matplotlib.pyplot as plt
from hmmlearn import hmm
from sklearn.metrics import confusion_matrix
import itertools
import os
import glob
import os.path as path

class HMMTrainer(object):
  def __init__(self, model_name='GaussianHMM', n_components=4, cov_type='diag', n_iter=1000):
    self.model_name = model_name
    self.n_components = n_components
    self.cov_type = cov_type
    self.n_iter = n_iter
    self.models = []
    if self.model_name == 'GaussianHMM':
      self.model = hmm.GaussianHMM(n_components=self.n_components,        covariance_type=self.cov_type,n_iter=self.n_iter)
    else:
      raise TypeError('Invalid model type')

  def train(self, X):
    np.seterr(all='ignore')
    self.models.append(self.model.fit(X))
    # Run the model on input data
  def get_score(self, input_data):
    return self.model.score(input_data)

def plot_confusion_matrix():

    classes = ["Cough","No Cough"]

    plt.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
    plt.colorbar()

    plt.title("Normalized Confusion Matrix")
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)

    plt.ylabel('True')
    plt.xlabel('Predicted')

    for i in range(2):
        for j in range(2):
            c = round(conf_matrix[j,i], 5)
            plt.text(i, j, str(c), va='center', ha='center')


hmm_models = []
input_folder = '/private/tmp/detector-de-tos-markov/entrenamiento'

X = np.array([])


# Parse the input directory
for filename in os.listdir(input_folder):
    # Read the input file
    filepath = os.path.join(input_folder, filename)
    sampling_freq, audio = wavfile.read(filepath)
    # Extract MFCC features
    mfcc_features = mfcc(audio, sampling_freq, nfft=1200)
    # Append to the variable X
    if len(X) == 0:
        X = mfcc_features
    else:
        X = np.append(X, mfcc_features, axis=0)
print('X.shape =', X.shape)
# Train and save HMM model
hmm_trainer = HMMTrainer(n_components=1)
print("Start training")
hmm_trainer.train(X)
print("Finish training")

input_folder_tos = '/private/tmp/detector-de-tos-markov/prueba'

real = []
predicted = []

TP = 0
FP = 0
FN = 0
TN = 0

max_score = -9999999999999999999

for dir in os.listdir(input_folder_tos):
    directory = os.path.join(input_folder_tos, dir)
    for filename in os.listdir(directory):
        filepath = os.path.join(directory, filename)
        sampling_freq, audio = wavfile.read(filepath)
        mfcc_features = mfcc(audio, sampling_freq, nfft=1200)
        score = hmm_trainer.get_score(mfcc_features)
        if score > max_score:
            max_score = score

for dir in os.listdir(input_folder_tos):
    directory = os.path.join(input_folder_tos, dir)
    for filename in os.listdir(directory):
        filepath = os.path.join(directory, filename)
        sampling_freq, audio = wavfile.read(filepath)
        mfcc_features = mfcc(audio, sampling_freq, nfft=1200)
        score = hmm_trainer.get_score(mfcc_features)
        if dir == "notos":
            real_output = "No Cough"
        else:
            real_output = "Cough"

        output = "Cough"

        if (score * 100) / max_score > 55:
            max_score = score
            output = "No Cough"
        predicted.append(output)
        real.append(real_output)

        print("\nPREDICTED::", output)
        print("ACTUAL_VALUE:", real_output)

        if output == "Cough" and real_output == "Cough":
            TP += 1
            print("-> TRUE POSITIVE")
        elif output == "Cough" and real_output == "No Cough":
            FN += 1
            print("-> FALSE NEGATIVE")
        elif output == "No Cough" and real_output == "No Cough":
            TN += 1
            print("-> TRUE NEGATIVE")
        elif output == "No Cough" and real_output == "Cough":
            FP += 1
            print("-> FALSE POSITIVE")

print()

conf_matrix = np.matrix([[TP/(TP+FP), FP/(TP+FP)], [FN/(FN+TN), TN/(FN+TN)]])
print("\n", conf_matrix)
plot_confusion_matrix()

TP = conf_matrix[0,0]
FP = conf_matrix[0,1]
FN = conf_matrix[1,0]
TN = conf_matrix[1,1]

acc = (TP + TN) / (TP + FP + FN + TN)
rec = TP / (TP + FN)
pre = TP / (TP + FP)
fms = (2 * rec * pre) / (rec + pre)

print('\nAccuracy =', acc)
print('Recall =', rec)
print('Presicion =', pre)
print('F-measure =', fms)

plt.show()