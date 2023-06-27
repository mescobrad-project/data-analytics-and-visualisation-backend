#!bin/bash/
sudo ps aux | grep Xorg >  my_display.txt
sudo chmod 777 my_display.txt
sudo grep -o "\S*" my_display.txt | grep -i "^:[0-9][0-9]" > actual_display.txt
sudo chmod 777 actual_display.txt
