clc
clear
close all

cam=webcam; %initalize a webcam object
count=0; %initalize count to 0
cam.Resolution = '640x480'; %set webcam's resolution to 640x480
while(count<500)
    img = cam.snapshot; % capture image
    imshow(img) %show image to user
    keep=input("Keep img? [y/n] ",'s'); % ask user to keep image
    if(keep=='y')
        imwrite(img,'Resistor'+string(count)+'.jpg'); %write image to file 
        count=count+1; % increase count
    else
        continue %continue and capture a new frame
    end
end

