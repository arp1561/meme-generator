import numpy as np
import cv2
import os, os.path
import pickle

training_image_array = np.zeros((1,122500))
output_array = np.zeros((1,6),float)

true_label= np.zeros((6,6),float)
for i in range(6):
    true_label[i,i] = 1


#anger dataset
i=0
DIR = 'dataset/anger'
n_anger = len([name for name in os.listdir(DIR) if os.path.isfile(os.path.join(DIR,name))])
print "Number of anger images = "+str(n_anger)
while i<n_anger:
    img = cv2.imread("dataset/anger/"+str(i)+".jpg")
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    unroll = gray.reshape(1,122500).astype(np.float32)
    training_image_array = np.vstack((training_image_array,unroll))
    output_array = np.vstack((output_array,true_label[0]))
    print i
    i+=1

print "Anger dataset loaded"

#fear dataset
i=0
DIR = 'dataset/fear'
n_fear = len([name for name in os.listdir(DIR) if os.path.isfile(os.path.join(DIR,name))])
print "Number of fear images = "+str(n_fear)
while i<n_fear:
    img = cv2.imread("dataset/fear/"+str(i)+".jpg")
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    unroll = gray.reshape(1,122500).astype(np.float32)
    training_image_array = np.vstack((training_image_array,unroll))
    output_array = np.vstack((output_array,true_label[1]))
    print i
    i+=1
print "Fear dataset loaded"

#happy dataset
i=0
DIR = 'dataset/happy'
n_happy = len([name for name in os.listdir(DIR) if os.path.isfile(os.path.join(DIR,name))])
print "Number of happy images = "+str(n_happy)
while i<n_happy:
    img = cv2.imread("dataset/happy/"+str(i)+".jpg")
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    unroll = gray.reshape(1,122500).astype(np.float32)
    training_image_array = np.vstack((training_image_array,unroll))
    output_array = np.vstack((output_array,true_label[2]))
    print i
    i+=1
print "Happy images loaded"


#neutral dataset
i=0
DIR = 'dataset/neutral'
n_neutral = len([name for name in os.listdir(DIR) if os.path.isfile(os.path.join(DIR,name))])
print "Number of neutral images = "+str(n_neutral)
while i<n_neutral:
    img = cv2.imread("dataset/neutral/"+str(i)+".jpg")
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    unroll = gray.reshape(1,122500).astype(np.float32)
    training_image_array = np.vstack((training_image_array,unroll))
    output_array = np.vstack((output_array,true_label[3]))
    print i
    i+=1
print "Neutral images loaded"


#sadness dataset
i=0
DIR = 'dataset/sadness'
n_sad = len([name for name in os.listdir(DIR) if os.path.isfile(os.path.join(DIR,name))])
print "Number of sad images = "+str(n_sad)
while i<n_sad:
    img = cv2.imread("dataset/sadness/"+str(i)+".jpg")
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    unroll = gray.reshape(1,122500).astype(np.float32)
    training_image_array = np.vstack((training_image_array,unroll))
    output_array = np.vstack((output_array,true_label[4]))
    print i
    i+=1
print "Sad images loaded"

#surprise dataset
i=0
DIR = 'dataset/surprise'
n_surprise = len([name for name in os.listdir(DIR) if os.path.isfile(os.path.join(DIR,name))])
print "Number of surprise images = "+str(n_surprise)
while i<n_surprise:
    img = cv2.imread("dataset/surprise/"+str(i)+".jpg")
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    unroll = gray.reshape(1,122500).astype(np.float32)
    training_image_array = np.vstack((training_image_array,unroll))
    output_array = np.vstack((output_array,true_label[5]))
    print i
    i+=1
print "Surprise images loaded"

with open('training_data.pkl','w') as f:
    pickle.dump(training_image_array,f)

with open('training_labels.pkl','w') as f:
    pickle.dump(output_array,f)




