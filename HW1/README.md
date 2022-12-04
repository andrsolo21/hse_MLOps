# hse_MLOps

1. Start venv
2. Create grpc files `python -m grpc_tools.protoc -I./grpc --python_out=. --grpc_python_out=. ./grpc/messages.proto`
3. Start REST API server: `uvicorn main:app --reload`
4. Start gRPC server: `server_grpc.py`

Swagger: http://127.0.0.1:8000/docs

Документация: http://127.0.0.1:8000/redoc