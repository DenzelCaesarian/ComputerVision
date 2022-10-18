import os
import cv2
import numpy as np
import face_recognition

# Mengambil Data dari Folder
path = 'ImagesData'
images = [] # Tempat menyimpan gambar
classNames = [] # Tempat menyimpan nama
myList = os.listdir(path) # Tempat menyimpan list gambar di dalam folder

print(myList)

for data in myList: # Looping untuk membaca data satu-persatu dari list
    currImg = cv2.imread(f'{path}/{data}')
    images.append(currImg) # Memasukkan gambar ke array images
    classNames.append(os.path.splitext(data)[0]) # Memasukkan nama ke array classNames

print(classNames)

# Encoding Images
def findEncoding(images):
    encodeList = []

    for image in images:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(image)[0]
        encodeList.append(encode)

    return encodeList

encodeKnown = findEncoding(images)

# Run on Unknown Data
fileName = 'ImagesTest/Unknown - 3.jpg'
unknownImage = face_recognition.load_image_file(fileName)

unknownImage = cv2.cvtColor(unknownImage, cv2.COLOR_BGR2RGB)
unknownImageLocations = face_recognition.face_locations(unknownImage)
unknownImageEncodings = face_recognition.face_encodings(unknownImage)

for encodeFace, (top, right, bottom, left) in zip(unknownImageEncodings, unknownImageLocations):
    name = 'UNKNOWN'

    matches = face_recognition.compare_faces(encodeKnown, encodeFace)
    distance = face_recognition.face_distance(encodeKnown, encodeFace)
    matchIndex = np.argmin(distance)

    if matches[matchIndex]:
        name = classNames[matchIndex].upper()
    
    cv2.rectangle(unknownImage, (left, top), (right, bottom), (0, 255, 0), 10)
    cv2.putText(unknownImage, name, (left, top - 20), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 5)

cv2.imshow(name, unknownImage)
cv2.waitKey(0)