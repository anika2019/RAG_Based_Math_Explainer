# Use a lightweight Python image
FROM python:3.11-slim

# Set the working directory inside the container
WORKDIR /app

# Copy your requirements first (helps with faster builds)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your app files
COPY . .

# Expose the port Streamlit uses
EXPOSE 7860

# Command to run your app on the port Hugging Face expects
CMD ["streamlit", "run", "main.py", "--server.port", "7860", "--server.address", "0.0.0.0"]