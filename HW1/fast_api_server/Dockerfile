# our base image
# docker build -f Dockerfile -t andrsolo21/fast_code ..
FROM python:3.10

WORKDIR /fast_code

COPY requirements.txt /fast_code/requirements.txt
COPY ../grpc/messages.proto /fast_code/grpc/messages.proto

RUN pip3 install -r /fast_code/requirements.txt
RUN python -m grpc_tools.protoc -I/fast_code/grpc --python_out=/fast_code --grpc_python_out=/fast_code /fast_code/grpc/messages.proto

COPY ../main.py /fast_code/main.py
COPY ../api_models.py /fast_code/api_models.py
COPY ../enum_models.py /fast_code/enum_models.py

# specify the port number the container should expose
EXPOSE 8000:8000

# run the application
#CMD ["uvicorn", "main:app", "--reload"]
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
#CMD ["python", "./server_grpc.py"]