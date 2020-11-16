from collections import namedtuple

EncoderParam = namedtuple('EncoderParam', ['text'])
encoder_param = EncoderParam([768, 256, 0.1, 'relu'])
HyperParam = namedtuple('HyperParam', ['encoder_param', 'lr'])
hyper_param = HyperParam(encoder_param, 5e-5)
