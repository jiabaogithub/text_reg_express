# -*- coding: utf-8 -*-

"""
flask初始化
"""
from logging.config import dictConfig

from flask import Flask
from flask_cors import CORS

from num_reg_resource import num

__author__ = 'ZhangJiabao'


def create_app(config_type):
    dictConfig({
        'version': 1,
        'formatters': {'default': {
            'format': '[%(asctime)s] %(name)s %(levelname)s in %(module)s %(lineno)d: %(message)s',
        }},
        'handlers': {'wsgi': {
            'class': 'logging.StreamHandler',
            'stream': 'ext://flask.logging.wsgi_errors_stream',
            'formatter': 'default'
        }},
        'root': {
            'level': 'DEBUG',
            # 'level': 'WARN',
            # 'level': 'INFO',
            'handlers': ['wsgi']
        }
    })
    # 加载flask配置信息
    app = Flask(__name__, static_folder='static', static_url_path='')
    # CORS(app, resources=r'/*',origins=['192.168.1.104'])  # r'/*' 是通配符，允许跨域请求本服务器所有的URL，"origins": '*'表示允许所有ip跨域访问本服务器的url
    CORS(app, resources={r"/*": {"origins": '*'}})  # r'/*' 是通配符，允许跨域请求本服务器所有的URL，"origins": '*'表示允许所有ip跨域访问本服务器的url
    app.config.from_object(config_type)
    app.register_blueprint(num, url_prefix='/num')

    # 初始化上下文
    ctx = app.app_context()
    ctx.push()

    return app


def loadProjContext():
    pass
