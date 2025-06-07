# qa-cosmetics-model
This is the server for answering cosmetics question.

# generate proto file
python -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. chat.proto

# run it
python -m uvicorn main:app --reload --port 8085

sudo docker run -d -p 8085:8085 qa-cosmetics:1.0  => API
sudo docker run -p 50051:50051 -d --name qa_service grpc_qa:1.0 => grpc
