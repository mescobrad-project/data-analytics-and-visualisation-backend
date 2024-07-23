#FROM vnmd/neurodesktop:2024-01-11
#FROM vnmd/neurodesktop:20220329
FROM vnmd/neurodesktop:20230112

#docker pull vnmd/neurodesktop:2024-01-12
RUN sudo apt update
RUN sudo apt -y install g++ make git qtbase5-dev-tools qtbase5-dev
RUN sudo git clone https://gitlab.com/Teuniz/EDFbrowser.git /home/user/EDFbrowser
RUN cd /home/user/EDFbrowser && sudo qmake
RUN cd /home/user/EDFbrowser && sudo make -j4
RUN cd /home/user/EDFbrowser && sudo make install

CMD sudo tail -f /dev/null
