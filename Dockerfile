FROM python:3.11-slim

WORKDIR /app

# Install OS-level dependencies needed by OpenCV and scikit-image
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency files first for layer caching
COPY requirements.txt pyproject.toml ./

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the package source
COPY finger_impedance/ ./finger_impedance/
COPY scripts/ ./scripts/
COPY extras/ ./extras/
COPY examples/ ./examples/

# Install the package in editable mode
RUN pip install --no-cache-dir -e .

# Default: drop into Python REPL with the package available
CMD ["python"]
