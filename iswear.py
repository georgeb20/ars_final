import cv2
# define a video capture object
cap = cv2.VideoCapture(0)  

cap.set(3, 640) # set the resolution
cap.set(4, 480)
cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
focus=0
while(True):
      
    # Capture the video frame
    # by frame
    focus = input("Enter a focus")
    cap.set(28,focus)
    ret, frame = cap.read()
  
    # Display the resulting frame
    cv2.imshow('frame', frame)
      
    # the 'q' button is set as the
    # quitting button you may use any
    # desired button of your choice
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
  
# After the loop release the cap object
# Destroy all the windows
cv2.destroyAllWindows()
