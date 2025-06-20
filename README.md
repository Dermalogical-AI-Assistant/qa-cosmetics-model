# qa-cosmetics-model
This is the server for answering cosmetics question.

# generate proto file
python -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. chat.proto

# run it
python -m uvicorn main:app --reload --port 8085

# build image
sudo docker build -t grpc_qa:1.0 -f Dockerfile_grpc .

sudo docker run -d -p 8087:8087 qa-cosmetics:1.0  => API
sudo docker run -p 50051:50051 -d --name qa_service grpc_qa:1.0 => grpc
