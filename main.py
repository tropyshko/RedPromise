import datetime
from tkinter import *

import cv2

faceCascade = cv2.CascadeClassifier("cascades/face.xml")
eyesCascade = cv2.CascadeClassifier("cascades/haarcascade_upperbody.xml")
eyes2Cascade = cv2.CascadeClassifier("cascades/haarcascade_lowerbody.xml")
font = cv2.FONT_HERSHEY_SIMPLEX
camera = 1
trainer = 'trainer/trainer.yml'
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('trainer/trainer.yml')
now = datetime.datetime.now()
date = now.strftime("%d-%m-%Y")
hours = now.strftime("%H")
minute = now.strftime("%M")
micro = now.microsecond
faceCascade = cv2.CascadeClassifier("cascades/haarcascade_frontalface_alt2.xml")
font = cv2.FONT_HERSHEY_SIMPLEX
id = 0
ids = 0
cam = cv2.VideoCapture(camera)
cam.set(3, 520)  # set video widht
cam.set(4, 540)  # set video height
minW = 0.1 * cam.get(3)
minH = 0.1 * cam.get(4)
f = open('config/data.xml')
lines = f.readlines()
count = int(str(lines[0]))
trained = str(lines[1])
training = str(lines[2])
f.close()
names = ['Face', 'Egor']
acce = ['Fce2', 'ACCESS ALLOWED']

def main():
	root = Tk()
	root.title("RedPromise")
	root.geometry("1600x900")

	def FaceDetectButton():
		face_detect()

	def FaceUpdateButton():
		face_dataset()


	btn = Button(text = "Распознавание лица", background = "#555", foreground = "#ccc",
	             padx = "40", pady = "8", font = "16", command = FaceDetectButton).place(x=75, y=20)

	btn = Button(text = "Обновление базы", background = "#555", foreground = "#ccc",
	             padx = "20", pady = "8", font = "16", command = FaceUpdateButton).place(x=75, y=80)

	btn = Button(text = "Выход", background = "#555", foreground = "#ccc",
	             padx = "40", pady = "8", font = "16", command = root.destroy)

	btn.pack()

	root.mainloop()

def face_recognition():

	while True:
		d = 0
		summ = 0
		medium = 0
		conf = 0
		id = "Unknown"
		access = "ACCESS DENIED"
		color = (0, 255, 255)
		match = 0
		ret, img = cam.read()
		gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		status = "DETECTING"
		trackers = []
		trackableObjects = {}

		faces = faceCascade.detectMultiScale(
			gray,
			scaleFactor = 1.2,
			minNeighbors = 5,
			minSize = (int(minW), int(minH)),
			)

		for (x, y, w, h) in faces:
			cv2.rectangle(img, (x, y), (x + w, y + h), (255 ,255, 255), 1)
			faceid = (gray[y:y + h, x:x + w])
			id, confidence = recognizer.predict(gray[y:y + h, x:x + w])
			min_match = 20
			status = ""
			if (confidence < 100):
				conf = round(100 - confidence)
				if (conf < min_match):
					id = "Unknown"
					access = "ACCESS DENIED"
					color = (0, 0, 255)
					ids = id
				else:
					access = acce[id]
					idents = "ID: {0}".format(id)
					ids = id
					id = names[id]
					d += 1
					summ += conf
					medium = round(summ / d)
					color = (0, 255, 0)
				match += 1

			now = datetime.datetime.now()

			def rect(img, x, y, w, h, text = ""):
				cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
			if id != "":
				l = len(format(id)) * 14

			cv2.rectangle(img, (x, y), (x + w, y + h), color, 1)                                    #Рамка вокруг лица
			cv2.rectangle(img, (x, y - 22), (x + l, y), color, -1)                                  #Фон имени
			cv2.putText(img, str(id), (x, y), font, 0.8, (0, 83, 138), 1, cv2.LINE_AA)                   #Имя
			cv2.putText(img, str(access), (x - 35, y + h + 30), font, 1, color, 1)                  #Доступ
			cv2.putText(img, 'ID: '+str(match), (x + 100, y + h - 5), font, 1, (255, 255, 0), 1)         #Процент

			# Вывод соответствия
			lineThickness = 10
			match_length = 300
			match_line = ((match_length // 100)*conf)
			cv2.putText(img, "MEDIUM: {0}".format(medium), (30, 120), font, 1, (255, 255, 255), 1)  #Среднее значение
			cv2.rectangle(img, (0, 45), (((match_length // 100)*min_match), 55), (0, 0, 255), 2)
			#Мин. линия
			cv2.rectangle(img, (0, 45), (match_length, 55), (0, 255, 0), 2)                         #Полная линия
			cv2.line(img, (0, 50), (match_line, 50), color, lineThickness)                          #Линия соответствия
			cv2.putText(img, str(conf)+'%', (x - 10, y + h - 5), font, 1, (255, 255, 0), 1)         #Процент



		cv2.putText(img, status, (70,270), font, 3, (0, 0, 255), 2)                  #Доступ
		cv2.imshow('Face Detection', img)
		k = cv2.waitKey(10) & 0xff  # Press 'ESC' for exiting video
		if k == 27:
			break
	__del__(cam)

def face_training():
	pass

def face_data():
	pass

def db():
	f = open('config/data.xml')
	lines = f.readlines()
	count = int(str(lines[0]))
	trained = str(lines[1])
	training = str(lines[2])
	f.close()

def __del__(cam):
	cv2.destroyAllWindows()
	cam.release()

def quit(self):
    self.destroy()
    exit()

if __name__ == "__main__":
	face_recognition()
