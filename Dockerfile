# What a Dockerfile does: Reads top to bottom to build an image.
# (a snapshot of your environment + code).

# Step 1 — Base image
# Starts from an official Python 3.11 image. slim means minimal OS —
# smaller image size, faster builds.
FROM python:3.11-slim

# Step 2 — Working directory
# All subsequent commands run from /app inside the container. 
# Keeps things organized.
WORKDIR /app

# Step 3 — Install dependencies first
# Copy requirements.txt and install before copying your code. 
# Docker caches each step — if your code changes but requirements don't, 
# it won't reinstall packages every time you rebuild.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Step 4 — Copy your code
# Now copy everything else into /app.
COPY . .

# Step 5 — Expose port and run
# EXPOSE documents which port the app uses. 
# CMD is the default command when the container starts.
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

