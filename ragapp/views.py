from django.shortcuts import render
# Assuming your chatbot logic is in a file named chat_logic.py within the ragapp directory
# from .chat_logic import get_chat_response
from ragapp.Rag import * 
# Create your views here.
from rest_framework.views import APIView
from rest_framework.response import Response
from .serializers import ChatSerializer
# from .your_chat_module import get_chat_response  # Import your chatbot function here

class ChatAPIView(APIView):
    def post(self, request):
        serializer = ChatSerializer(data=request.data)
        if serializer.is_valid():
            user_message = serializer.validated_data['message']
            # chat_response = get_chat_response(user_message)  # Call your chatbot logic
            
            return Response({'message': user_message, 'response': resp(user_message)})
        return Response(serializer.errors, status=400)
