FROM freesurfer/freesurfer:7.1.1
COPY license.txt /usr/local/freesurfer/.license


RUN yum -y install openssh-server openssh-clients
#RUN #systemctl start sshd

#RUN apt-get update
#
#RUN apt-get install openssh-server
#
#RUN useradd -rm -d /home/ubuntu -s /bin/bash -g root -G sudo -u 1000 test
#
#RUN  echo 'test:test' | chpasswd
#
#RUN #service sshd start

EXPOSE 22

CMD ["/usr/sbin/sshd","-D"]
CMD tail -f /dev/null
