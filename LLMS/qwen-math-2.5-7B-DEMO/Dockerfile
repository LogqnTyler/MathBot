FROM python:3.11-slim

WORKDIR /app

# TIRMathAgent's code backend spins up a Jupyter kernel and executes the
# model's generated Python here, inside this container (container = sandbox).
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY demo.py .

CMD ["python", "demo.py"]
