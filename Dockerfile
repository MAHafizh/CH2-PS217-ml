# Use an image that includes necessary development libraries
FROM python:3.8

# Install system dependencies
RUN apt-get update && apt-get install -y libblas-dev liblapack-dev gfortran

# Set the working directory
WORKDIR /app

# Copy the rest of your application files
COPY . /app

# Install Python packages
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

EXPOSE 8080

# Define the command to run your application
CMD ["gunicorn", "-b", "0.0.0.0:8080", "app:app"]