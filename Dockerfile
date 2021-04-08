FROM ubuntu:latest

RUN apt update && apt -y upgrade
RUN apt install -y python3 python3-dev python3-pip

RUN mkdir thalamocortical
WORKDIR thalamocortical/
COPY . .

RUN pip3 install --upgrade -r requirements.txt

EXPOSE 8899
CMD ["jupyter", "lab", "--port=8899", "--no-browser", "--ip=0.0.0.0", "--allow-root"]
