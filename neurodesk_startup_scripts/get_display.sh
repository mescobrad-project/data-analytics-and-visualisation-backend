#!bin/bash/
ps aux | grep Xorg >  my_display.txt
grep -o "\S*" my_display.txt | grep -i "^:[0-9][0-9]" > actual_display.txt