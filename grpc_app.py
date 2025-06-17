import chat_pb2
import chat_pb2_grpc
from answer import get_answer
from datetime import datetime
import grpc
import logging
import traceback
from concurrent import futures

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ChatServicer(chat_pb2_grpc.ChatServiceServicer):
    def SendMessage(self, request, context):
        try:
            # Input validation
            if not request.question:
                context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
                context.set_details("Question cannot be empty")
                return chat_pb2.MessageResponse(success=False)
            
            logger.info(f"Processing question: {request.question}")
            
            # Business logic
            answer = get_answer(question=request.question)
            
            return chat_pb2.MessageResponse(
                success=True,
                answer=answer,
                timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            )
            
        except Exception as e:
            logger.error(f"Error processing request: {traceback.format_exc()}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f"Server error: {str(e)}")
            return chat_pb2.MessageResponse(success=False)

def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    chat_pb2_grpc.add_ChatServiceServicer_to_server(ChatServicer(), server)
    server.add_insecure_port('0.0.0.0:50051')
    server.start()
    print("Python gRPC server running on port 50051")
    server.wait_for_termination()

if __name__ == '__main__':
    serve()