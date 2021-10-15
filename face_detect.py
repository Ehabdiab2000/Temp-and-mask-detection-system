import sys

from PyQt5 import QtWidgets, QtGui
from PyQt5.QtCore import QSize, QTimer
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QApplication, QDialog, QMainWindow, QTableWidgetItem
from PyQt5.uic import loadUi
# import the necessary packages
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import VideoStream
import numpy as np
import imutils
import time
import datetime
import cv2
import os
from playsound import playsound
import sqlite3

class MainDialog(QDialog):
    def __init__(self):
        super(MainDialog, self).__init__()
        loadUi('life2coding.ui', self)
        self.mask = False
        self.mask2 = False
        self.image = None
        self.frame = None
        self.image2 = None
        self.frame2 = None
        self.croped = None
        self.croped2 = None
        self.start_webcam()
        self.load.clicked.connect(self.loadLog)
        self.prototxtPath = r"face_detector\deploy.prototxt"
        self.weightsPath = r"face_detector\res10_300x300_ssd_iter_140000.caffemodel"
        self.faceNet = cv2.dnn.readNet(self.prototxtPath, self.weightsPath) # model to detect faces locations

        # load the face mask detector model from disk  to detect if wearink mask or no
        self.maskNet = load_model("mask_detector.model")

    def start_webcam(self):
        # self.capture=cv2.VideoCapture(0) #0 =default #1,2,3 =Extra Webcam
        # self.capture.set(cv2.CAP_PROP_FRAME_WIDTH,480)
        # self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT,640)
        self.vs = VideoStream(src=0).start()
        self.vs2 = VideoStream(src=1).start()
        self.frame = self.vs.read()
        self.frame = imutils.resize(self.frame, width=400)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(100)

    def update_frame(self):
        self.frame = self.vs.read()
        self.frame = cv2.flip(self.frame, 1)
        self.detected_image = self.detect_face(self.frame)

        self.displayImage(self.detected_image)

        self.frame2 = self.vs2.read()
        self.frame2 = cv2.flip(self.frame2, 1)
        self.detected_image2 = self.detect_face2(self.frame2)

        self.displayImage2(self.detected_image2)
    def detect_face(self, frame):
        # grab the dimensions of the frame and then construct a blob
        # from it
        (h, w) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(frame, 1.0, (224, 224),
                                     (104.0, 177.0, 123.0))

        # pass the blob through the network and obtain the face detections
        self.faceNet.setInput(blob)
        detections = self.faceNet.forward()
        print(detections.shape)

        # initialize our list of faces, their corresponding locations,
        # and the list of predictions from our face mask network
        faces = []
        locs = []
        preds = []

        # loop over the detections
        for i in range(0, detections.shape[2]):
            # extract the confidence (i.e., probability) associated with
            # the detection
            confidence = detections[0, 0, i, 2]

            # filter out weak detections by ensuring the confidence is
            # greater than the minimum confidence
            if confidence > 0.5:
                # compute the (x, y)-coordinates of the bounding box for
                # the object
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")

                # ensure the bounding boxes fall within the dimensions of
                # the frame
                (startX, startY) = (max(0, startX), max(0, startY))
                (endX, endY) = (min(w - 1, endX), min(h - 1, endY))

                # extract the face ROI, convert it from BGR to RGB channel
                # ordering, resize it to 224x224, and preprocess it
                face = frame[startY:endY, startX:endX]
                face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                face = cv2.resize(face, (224, 224))
                face = img_to_array(face)
                face = preprocess_input(face)

                # add the face and bounding boxes to their respective
                # lists
                faces.append(face)
                locs.append((startX, startY, endX, endY))

        # only make a predictions if at least one face was detected
        if len(faces) > 0:
            # for faster inference we'll make batch predictions on *all*
            # faces at the same time rather than one-by-one predictions
            # in the above `for` loop
            faces = np.array(faces, dtype="float32")
            preds = self.maskNet.predict(faces, batch_size=1)
            # print(preds)

        # display the label and bounding box rectangle on the output
        # frame
        for (box, pred) in zip(locs, preds):
            # unpack the bounding box and predictions
            (startX, startY, endX, endY) = box
            (mask, withoutMask) = pred

            # determine the class label and color we'll use to draw
            # the bounding box and text
            label = "Mask" if mask > withoutMask else "No Mask"
            color = (0, 255, 0) if label == "Mask" else (0, 0, 255)
            if mask > withoutMask:
                self.mask = True
            else:
                self.mask = False
            # self.mask = True if mask > withoutMask else False
            # include the probability in the label
            label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)

            # display the label and bounding box rectangle on the output
            # frame
            cv2.putText(frame, label, (startX, startY - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
            cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
            x = 20
            self.croped = frame[startY:endY, startX:endX].copy()
            temp = self.get_temp()
            ## save the cropped face image
            # filename = str(time.strftime("%Y %m %d %H %M %S")).replace(':', ' ')
            # cv2.imwrite("C:\stdprojcts\photos\{0}.jpg".format(filename), self.croped)
            if self.mask == True and 35 < temp < 37:
                self.Label.setStyleSheet("background-color: green")
                self.Label.setText('Proceed to the Counter')

            elif self.mask == False:
                    self.Label.setStyleSheet("background-color: red")
                    self.Label.setText('Please Wear A Face Mask')
            else:
                self.Label.setStyleSheet("background-color: red")
                self.Label.setText('Please to the Security to check your Temperature')
            self.insertlog(self.croped ,self.mask,temp)
        # cv2.imshow("Frame", self.croped)
        # return frame with detection
        return frame

    def insertlog(self, img,mask,temp,fine='NO'):
        time_ = str(time.strftime("%Y %m %d %H %M "))
        print(time_)
        filename = time_.replace(':', ' ')
        print(filename)
        cv2.imwrite("photos\{0}.jpg".format(filename), img)
        db = sqlite3.connect('faceMask.db')
        cursor = db.cursor()
        row = (time_, str(mask),temp, time_,fine)
        command = '''REPLACE INTO log (time ,mask,temp,photo,fine) VALUES (?,?,?,?,?)'''
        cursor.execute(command, row)
        db.commit()

    def loadLog(self):
        format = "yyyy MM dd hh mm";
        print(self.fromTime.dateTime().toString(format))
        db = sqlite3.connect('faceMask.db')
        cursor = db
        command = '''SELECT * FROM log WHERE time BETWEEN ? AND ? '''
        row = (self.fromTime.dateTime().toString(format), self.toTime.dateTime().toString(format))
        result = cursor.execute(command, row)

        ### to Fill table with result
        self.table.setRowCount(0)
        for row_number, row_data in enumerate(result):
            self.table.insertRow(row_number)
            for column_number, data in enumerate(row_data):
                if (column_number == 4):
                    item = self.getImageLable(data)
                    self.table.setCellWidget(row_number, column_number, item)
                else:
                    self.table.setItem(row_number, column_number, QTableWidgetItem(str(data)))
        self.table.verticalHeader().setDefaultSectionSize(80)

    def getImageLable(self, imName):
        imageLabel = QtWidgets.QLabel(self.log)
        imageLabel.setText("aa")
        imageLabel.setScaledContents(True)
        # pixmap = QtGui.QPixmap("2020 11 16 17 26 04.jpg")
        imagename = str("photos\{0}.jpg".format(str(imName)))
        print(imagename)

        pixmap = QtGui.QPixmap(str("photos\{0}.jpg".format(imName)))

        # pixmap.loadFromData(image,'jpg')
        imageLabel.setPixmap(pixmap)
        return imageLabel

    def detect_face2(self, frame):
        # grab the dimensions of the frame and then construct a blob
        # from it
        (h, w) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(frame, 1.0, (224, 224),
                                     (104.0, 177.0, 123.0))

        # pass the blob through the network and obtain the face detections
        self.faceNet.setInput(blob)
        detections = self.faceNet.forward()
        print(detections.shape)

        # initialize our list of faces, their corresponding locations,
        # and the list of predictions from our face mask network
        faces = []
        locs = []
        preds = []

        # loop over the detections
        for i in range(0, detections.shape[2]):
            # extract the confidence (i.e., probability) associated with
            # the detection
            confidence = detections[0, 0, i, 2]

            # filter out weak detections by ensuring the confidence is
            # greater than the minimum confidence
            if confidence > 0.5:
                # compute the (x, y)-coordinates of the bounding box for
                # the object
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")

                # ensure the bounding boxes fall within the dimensions of
                # the frame
                (startX, startY) = (max(0, startX), max(0, startY))
                (endX, endY) = (min(w - 1, endX), min(h - 1, endY))

                # extract the face ROI, convert it from BGR to RGB channel
                # ordering, resize it to 224x224, and preprocess it
                face = frame[startY:endY, startX:endX]
                face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                face = cv2.resize(face, (224, 224))
                face = img_to_array(face)
                face = preprocess_input(face)

                # add the face and bounding boxes to their respective
                # lists
                faces.append(face)
                locs.append((startX, startY, endX, endY))

        # only make a predictions if at least one face was detected
        if len(faces) > 0:
            # for faster inference we'll make batch predictions on *all*
            # faces at the same time rather than one-by-one predictions
            # in the above `for` loop
            faces = np.array(faces, dtype="float32")
            preds = self.maskNet.predict(faces, batch_size=1)
            # print(preds)

        # display the label and bounding box rectangle on the output
        # frame
        for (box, pred) in zip(locs, preds):
            # unpack the bounding box and predictions
            (startX, startY, endX, endY) = box
            (mask, withoutMask) = pred

            # determine the class label and color we'll use to draw
            # the bounding box and text
            label = "Mask" if mask > withoutMask else "No Mask"
            color = (0, 255, 0) if label == "Mask" else (0, 0, 255)
            if mask > withoutMask:
                self.mask2 = True
            else:
                self.mask2 = False
             # include the probability in the label
            label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)

            # display the label and bounding box rectangle on the output
            # frame
            cv2.putText(frame, label, (startX, startY - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
            cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
            x = 20
            self.croped2 = frame[startY:endY, startX:endX].copy()

            # ## save the cropped face image
            # filename = str(time.strftime("%Y %m %d %H %M %S")).replace(':', ' ')
            # cv2.imwrite("C:\stdprojcts\photos\{0}.jpg".format(filename), self.croped)
            if self.mask2 == True :
                fine ='NO'
            else:
                fine ='500'
            self.insertlog(self.croped2, self.mask2, None,fine)
        # return frame with detection
        return frame


    def displayImage(self, img):
        qformat = QImage.Format_Indexed8

        if len(img.shape) == 3:  # rows[0],cols[1],channels[2]
            if (img.shape[2]) == 4:
                qformat = QImage.Format_RBA8888G
            else:
                qformat = QImage.Format_RGB888
        img = QImage(img, img.shape[1], img.shape[0], img.strides[0], qformat)
        # BGR > RGB
        img = img.rgbSwapped()
        self.imgLabel.setPixmap(QPixmap.fromImage(img))
        self.imgLabel.setScaledContents(True)


    def displayImage2(self, img):
        qformat = QImage.Format_Indexed8

        if len(img.shape) == 3:  # rows[0],cols[1],channels[2]
            if (img.shape[2]) == 4:
                qformat = QImage.Format_RBA8888G
            else:
                qformat = QImage.Format_RGB888
        img = QImage(img, img.shape[1], img.shape[0], img.strides[0], qformat)
        # BGR > RGB
        img = img.rgbSwapped()
        self.imgLabel2.setPixmap(QPixmap.fromImage(img))
        self.imgLabel2.setScaledContents(True)





    def service(self):
        #print(self.mask)
        #print(self.get_temp())
        filename = str(time.strftime("%Y %m %d %H %M %S")).replace(':',' ')

        #print(filename)

        cv2.imwrite("C:\stdprojcts\photos\{0}.jpg".format(filename), self.croped)
        if self.mask == True and 35 < self.get_temp() < 37:
            playsound('Recording 3.mp3')

        else:

            playsound('Recording.mp3')

    def get_temp(self):

        return 36

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainDialog()
    window.setWindowTitle('Face Mask Detection')
    window.show()
    sys.exit(app.exec_())
