# -*- coding: utf-8 -*-
# @Time    : 2019/8/23 21:55
# @Author  : zhoujun
from .model import Model
from .loss import PANLoss


def get_model(config):
    return Model(config)

def get_loss(config):
    alpha = config['alpha']
    beta = config['beta']
    delta_agg = config['delta_agg']
    delta_dis = config['delta_dis']
    ohem_ratio = config['ohem_ratio']
    return PANLoss(alpha=alpha, beta=beta, delta_agg=delta_agg, delta_dis=delta_dis, ohem_ratio=ohem_ratio)
