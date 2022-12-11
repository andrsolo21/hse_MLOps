# our base image
# docker build -f Dockerfile -t andrsolo21/grpc_server_code ..
FROM python:3.10

WORKDIR /grpc_code

COPY ../requirements.txt /grpc_code/requirements.txt
COPY ../grpc/messages.proto /grpc_code/grpc/messages.proto

RUN pip3 install -r /grpc_code/requirements.txt
RUN python -m grpc_tools.protoc -I/grpc_code/grpc --python_out=/grpc_code --grpc_python_out=/grpc_code /grpc_code/grpc/messages.proto

COPY ../server_grpc.py /grpc_code/server_grpc.py
COPY ../ml_models.py /grpc_code/ml_models.py
COPY ../api_models.py /grpc_code/api_models.py
COPY ../enum_models.py /grpc_code/enum_models.py

# specify the port number the container should expose
EXPOSE 50051:50051

# run the application
CMD ["python", "server_grpc.py"]