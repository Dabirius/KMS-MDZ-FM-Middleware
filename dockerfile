# start by pulling the python image
FROM python:3.10

# copy the requirements file into the image
COPY ./requirements.txt /app/requirements.txt

# switch working directory
WORKDIR /app

# install the dependencies and packages in the requirements file
RUN apt-get update && apt-get install -y sqlite3 libsqlite3-dev && sqlite3 --version && pip install --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Make port 5000 available to the world outside this container
EXPOSE 5000

# copy every content from the local file to the image
COPY . /app

# configure the container to run in an executed manner
ENTRYPOINT [ "python3" ]

CMD ["middleware.py" ]