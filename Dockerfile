#Use official Python image
FROM python:3.12.8-slim

#Set the working Directory
WORKDIR /app

#copy all the files to container
COPY . .

#Install Dependencies
RUN pip install --no-cache-dir -r requirements.txt

#Expose the port the app runs on
EXPOSE 5000

RUN ls -lrt


#COMMAND TO RUN FLASK APP
CMD ["python", "app/app.py"]

