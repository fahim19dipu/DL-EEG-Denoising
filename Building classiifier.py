# -*- coding: utf-8 -*-
"""
Created on Thu Aug 18 20:04:04 2022

@author: user
"""
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy import io, signal
from scipy.fft import rfft, rfftfreq#, fft, fftfreq, fftshift
from mne.decoding import CSP
import mne
from sklearn.preprocessing import LabelEncoder,StandardScaler

from sklearn.model_selection import train_test_split#,cross_val_score
from sklearn import metrics#,svm
import tensorflow as tf
import ntpath
import novel_cnn as novel
import itertools
""" Plot confussion matrix and save as image"""
def plot_confusion_matrix(cm,target_names,title='Confusion matrix',cmap=None, normalize=None):
    
    accuracy = np.trace(cm) / np.sum(cm).astype('float')
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    #return plt
    plt.tight_layout()
    plt.show()
    #plt.savefig('images/cf.png',pad_inches=0.1)

def path_leaf(path):
    head, tail = ntpath.split(path)
    return tail or ntpath.basename(head)

def evaluation(model,test_x,test_y):
    lebels = ["EEG","EOG"]
    try:
        test_loss,test_accuracy = model.evaluate(test_x, test_y)
        print("Test acciracy ",test_accuracy)
        y_pred=model.predict(test_x)
        print("Confussion matrix\n", metrics.confusion_matrix(test_y, np.argmax(y_pred,axis=1)))
        plot_confusion_matrix(metrics.confusion_matrix(test_y, np.argmax(y_pred,axis=1)), lebels)
        #Classofocation report
        print("Classification report \n",metrics.classification_report
              (test_y, np.argmax(y_pred,axis=1),target_names=lebels))
    except:
        test_accuracy = model.score(test_x, test_y)
        print("Test acciracy ",test_accuracy)
        y_pred=model.predict(test_x)
        print("Confussion matrix\n", metrics.confusion_matrix(test_y,y_pred))
        plot_confusion_matrix(metrics.confusion_matrix(test_y,y_pred), lebels)
        #Classofocation report
        print("Classification report \n",metrics.classification_report(test_y, y_pred,target_names=lebels))
 
def fft_gen(y):
    N = 256*2
    fs =256
    yf = rfft(y)
    #print(yf.shape)
    xf = rfftfreq(N, 1 / fs)
    #print(xf.shape)
    plt.plot(xf, np.abs(yf[1]))
    plt.show()
    return np.abs(rfft(y))
def spectogram_gen(mat):
    fs = 256
    f, t, spec= signal.spectrogram(mat,fs) 
    plt.pcolormesh(t, f,spec[1], shading='gouraud')
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.show() 
    return spec
def stft_gen(y):
    return np.abs(mne.time_frequency.stft(y, wsize=4, tstep=None, verbose=None))

def leb_encoding(Y):
    le = LabelEncoder()
    le.fit(Y)
    Y = le.transform(Y)
    return Y
def input_encoding(X):
    ss = StandardScaler()
    ss.fit(X)
    X = ss.transform(X)
    return X

def load_data(filepath):
    mat = io.loadmat(filepath,struct_as_record=True, squeeze_me=False)
    filename = path_leaf(filepath)
    filename, file_extension = os.path.splitext(filename)
    
    mat=np.array(mat[filename])                 # Signal value (time domain)
    fft_sig = fft_gen(mat)                      # FFT value    (Frequency donain)
    stft = stft_gen(mat)                        # Short time furier transform value(Frequency donain)                      
    spec = spectogram_gen(mat)                  # Spectogram value

    return mat,spec,stft,fft_sig


# creating the dataset variable
class_size =3400
""" initializing dataset shapes"""
X_sig=np.zeros(0).reshape(0,512)                # Signal value (time domain)
X_spec=np.zeros(0).reshape(0,129,2)             # Spectogram value
X_stft= np.zeros(0).reshape(0,3,256)            # Short time furier transform value(Frequency donain)
X_fft = np.zeros(0).reshape(0,257)              # FFT value    (Frequency donain)
Y=np.zeros(0)


EOG, spec_eog, stft_eog, fft_eog = load_data('F:/1 Projects/datasets/eeg denoise/EOG_all_epochs.mat') 
#EOG = EOG[:class_size]

leb = np.full((len(EOG)), int(1))                # adding lebel
X_sig=np.append(X_sig,EOG,axis=0)
X_spec=np.append(X_spec,spec_eog,axis=0)
X_stft = np.append(X_stft,stft_eog,axis=0)
X_fft = np.append(X_fft,fft_eog,axis=0)
Y =  np.append(Y,leb,axis=0)

#EEG = EEG[:class_size]  

EEG,spec_eeg , stft_eeg, fft_eeg= load_data('F:/1 Projects/datasets/eeg denoise/EEG_all_epochs.mat') 
EEG,spec_eeg , stft_eeg, fft_eeg = EEG[:class_size], spec_eeg[:class_size] , stft_eeg[:class_size], fft_eeg[:class_size]
X_sig=np.append(X_sig,EEG,axis=0)
X_spec=np.append(X_spec,spec_eeg,axis=0)
X_stft = np.append(X_stft,stft_eeg,axis=0)
X_fft = np.append(X_fft,fft_eeg,axis=0)
leb = np.full((len(EEG)), int(0))
Y =  np.append(Y,leb,axis=0)

# n_of_comp = 6
# csp_alt= CSP(n_of_comp,reg=None, log=None, norm_trace=False)
#print("csp")
#ft_spec=csp_alt.fit_transform(X_alt,Y)

X = X_fft
# X = X_stft.transpose(0, 2, 1)
# X = X_spec
# X= X_sig
try:
    X = input_encoding(X)
    Y = leb_encoding(Y)
except:
    Y = leb_encoding(Y)
X = X[...,np.newaxis]    
print("Sample dataset shape",X.shape)
train_x, test_x, train_y, test_y = train_test_split(X, Y, test_size = 0.25, 
                                                    shuffle=True,random_state=42)
print("Traing set shape",train_x.shape)
print("Testing set shape",test_x.shape)
print("start training")

"""                                    CNN                        """
# try:
#     model = novel.Novel_CNN(input_size=(train_x.shape[1],train_x.shape[2]))
# except:
#     model = novel.Novel_CNN(input_size=(train_x.shape[1],1))
    
model = novel.RNN_lstm(257)
model.compile(tf.keras.optimizers.Adam(learning_rate=0.001),
              loss="sparse_categorical_crossentropy",
              metrics=['accuracy'])
History = model.fit(train_x, train_y, epochs=2,batch_size=32,validation_split=.1,shuffle=True) 
evaluation(model, test_x, test_y)


x =X[1]
x = x[np.newaxis,...] 
print(x.shape)
predictions = model.predict(x)
print(predictions)
if np.argmax(predictions)==1:
    print("EOG")
else:
    print("EEG")
model.save("saved_model/my_model.h5")