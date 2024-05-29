from django.urls import path
from .views import ChatAPIView

urlpatterns = [
    path('api/chat/', ChatAPIView.as_view(), name='chat-api'),
]
