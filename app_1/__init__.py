from flask import Flask

app = Flask(__name__)
app.config.from_object("config")

from app_1 import views
