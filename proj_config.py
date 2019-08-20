# -*- coding: utf-8 -*-

"""
本模块是Flask的配置模块
"""
import os

__author__ = 'ZhangJiabao'

basedir = os.path.abspath(os.path.dirname(__file__))


class BaseConfig:  # 基本配置类
    # SECRET_KEY = os.getenv('SECRET_KEY', b'\xe4r\x04\xb5\xb2\x00\xf1\xadf\xa3\xf3V\x03\xc5\x9f\x82$^\xa25O\xf0R\xda')
    SECRET_KEY = b'\xe4r\x04\xb5\xb2\x00\xf1\xadf\xa3\xf3V\x03\xc5\x9f\x82$^\xa25O\xf0R\xda'
    JSONIFY_MIMETYPE = 'application/json; charset=utf-8'  # 默认JSONIFY_MIMETYPE的配置是不带'; charset=utf-8的'
    JSON_AS_ASCII = False  # 若不关闭，使用JSONIFY返回json时中文会显示为Unicode字符
    ENCODING = 'utf-8'

    # 自定义的配置项
    ADDRESS_LABELS = ["COUNTY", "STREET", "COMMUNITY", "ROAD", "NUM", "POI", "CITY", "VILLAGE"]
    PERSON_LABELS = ["TIME", "LOCATION", "PERSON_NAME", "ORG_NAME", "COMPANY_NAME", "PRODUCT_NAME"]

    # MAX_CONTENT_LENGTH = 5242880 # 5m
    # MAX_CONTENT_LENGTH = 1048576 #1m
    # MAX_CONTENT_LENGTH = 524288 #512k


class DevelopmentConfig(BaseConfig):
    ENV = 'development'
    DEBUG = True


class TestingConfig(BaseConfig):
    TESTING = True
    WTF_CSRF_ENABLED = False


class ProductionConfig(BaseConfig):
    DEBUG = False


config = {
    'testing': TestingConfig,
    'default': DevelopmentConfig
    # 'default': ProductionConfig
}
