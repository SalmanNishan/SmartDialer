FROM python:3.10.12
RUN apt-get update && apt-get install -y supervisor

# Set the working directory inside the container
WORKDIR /app

# Copy the application files
COPY . .

# Install the required Python packages
RUN pip install -r /app/requirements.txt

# Copy the supervisord configuration file
COPY supervisord.conf /etc/supervisor/conf.d/supervisord.conf

# Expose the necessary ports
EXPOSE 50051 50052 50053 50054 50055 50056 50057 50058

# Start supervisord
CMD ["/usr/bin/supervisord"]
