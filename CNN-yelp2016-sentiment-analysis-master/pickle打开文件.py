#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author :Gongqin
# @Time ：2018/7/9 11:06
# @File :pickle.py
import pickle
f = open('E:/project/CNN-yelp-challenge-2016-sentiment-classification-master/CNN-yelp-challenge-2016-sentiment-classification-master/399850by50reviews_words_index.pkl','rb')
info = pickle.load(f)
print(info)