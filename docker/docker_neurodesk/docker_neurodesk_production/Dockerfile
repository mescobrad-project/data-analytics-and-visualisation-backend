#FROM vnmd/neurodesktop:20220329
FROM vnmd/neurodesktop:20230112

RUN sudo apt update
RUN sudo apt -y install g++ make git qtbase5-dev-tools qtbase5-dev
RUN git clone https://gitlab.com/Teuniz/EDFbrowser.git /home/user/EDFbrowser
RUN cd /home/user/EDFbrowser && qmake
RUN cd /home/user/EDFbrowser && make -j4
RUN cd /home/user/EDFbrowser && sudo make install

CMD tail -f /dev/null
