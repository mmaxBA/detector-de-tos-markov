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

def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

hmm_models = []
input_folder = 'C:\\Users\\max29\\Downloads\\tos\\entrenamiento'

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



input_folder_tos = './detector-de-tos-markov'


real = []
predicted = []

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
            print(score)
            real_output = "No Cough"
        else:
            real_output = "Cough"
        output = "Cough"
        if (score * 100) / max_score > 55:
            max_score = score
            output = "No Cough"
        predicted.append(output)
        real.append(real_output)

cm = confusion_matrix(real, predicted)
np.set_printoptions(precision=2)
classes = ["Cough","No Cough"]
plt.figure()
plot_confusion_matrix(cm, classes=classes, normalize=True, title='Normalized confusion matrix')

plt.show()
