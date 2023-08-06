FROM tensorflow/tensorflow:2.13.0-gpu
LABEL authors="Rubén Ballester"

WORKDIR /app
# Copy the project to the container.
COPY . .
#Update pip
RUN pip install --upgrade pip
# Install giotto-ph. This is not a trivial task because it has not been published in PyPI for Python 3.11. We use
# the forked version of the project included in the repository.
RUN apt update -y
RUN apt install -y build-essential libssl-dev cmake git
WORKDIR /app/giotto-ph
RUN pip install .
# Go again to the project folder.
WORKDIR /app
# Install the rest of requirements of the project.
RUN pip install -r requirements.txt
# Set the entrypoint of the container. We execute the main.py script.
# The script will be executed with the arguments passed to the docker run command.
ENTRYPOINT ["python", "main.py"]