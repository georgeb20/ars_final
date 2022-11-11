from skimage.metrics import structural_similarity as compare_ssim
import cv2



cap = cv2.VideoCapture(0)
while cap.isOpened():
    ret,frame1 = cap.read()

    ret, frame2 = cap.read()

    if not ret:
        break
    grayA = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    grayB = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    (score) = compare_ssim(grayA, grayB, full=True)
    if(score<.90):
       print("SSIM: {}".format(score))



