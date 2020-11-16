from collections import namedtuple
from constants import VISUAL_DIM


EncoderParam = namedtuple('EncoderParam', ['text', 'visual', 'audio'])
InteractParam = namedtuple('InteractParam', ['visual', 'audio'])

encoder_param = EncoderParam([768, 256],
                             [VISUAL_DIM, 1, 256, 0.1, 'relu', 1],
                             [74, 1, 256, 0.1, 'relu', 1])

interact_param = InteractParam([256, 256, 4, 1024, 0.1, 'relu'],
                               [256, 256, 4, 1024, 0.1, 'relu'])

HyperParam = namedtuple('HyperParam', ['encoder_param', 'interact_param', 'num_interactions', 'lr'])
hyper_param = HyperParam(encoder_param, interact_param, 3, 1e-5)
