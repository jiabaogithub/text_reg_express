# -*- coding: utf-8 -*-

"""
flask 入口
"""
import os

import proj_config as cf
from flaskr import create_app, loadProjContext

__author__ = 'ZhangJiabao'

from flask import jsonify, make_response, redirect



# 加载flask配置信息
# app = create_app('config.DevelopmentConfig')
app = create_app(cf.config['default'])
# 加载项目上下文信息
loadProjContext()


@app.errorhandler(404)
def not_found(error):
    return make_response(jsonify({'error': 'Not found'}), 404)


@app.errorhandler(400)
def not_found(error):
    return make_response(jsonify({'error': '400 Bad Request,参数或参数内容异常'}), 400)


@app.route('/')
def index_sf():
    # return render_template('index.html')
    return redirect('index.html')


# nohup python -u proj_main.py >log.out 2>&1 &
if __name__ == '__main__':
    if os.name == 'nt':  # windows path config
        ip = '192.168.1.107'
    else:  # linux path config
        ip = '192.168.1.111'
    app.run(ip, 5009, app, use_reloader=False)
    # app.run('localhost', 5006, app, use_reloader=False)
