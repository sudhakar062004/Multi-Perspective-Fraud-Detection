from django.urls import path

from . import views

urlpatterns = [path("", views.index, name="index"),
               path("UserLogin.html", views.UserLogin, name="UserLogin"),	      
               path("UserLoginAction", views.UserLoginAction, name="UserLoginAction"),
               path("LoadDataset.html", views.LoadDataset, name="LoadDataset"),
	       path("LoadDatasetAction", views.LoadDatasetAction, name="LoadDatasetAction"),
               path("ProcessMining", views.ProcessMining, name="ProcessMining"),
               path("RunML", views.RunML, name="RunML"),
               path("DetectFraud.html", views.DetectFraud, name="DetectFraud"),
	       path("DetectFraudAction", views.DetectFraudAction, name="DetectFraudAction"),	       
]
