from .utils import count_trainable_params

_MODELS = dict()


def register(name):

  def add_to_dict(fn):
    global _MODELS
    _MODELS[name] = fn
    return fn

  return add_to_dict


def get_models(hparams, summary):
  if hparams.model not in _MODELS:
    print('models {} not found'.format(hparams.model))
    exit()

  generator, discriminator = _MODELS[hparams.model](hparams)

  summary.scalar('model/trainable_parameters/generator',
                 count_trainable_params(generator))
  summary.scalar('model/trainable_parameters/discriminator',
                 count_trainable_params(discriminator))

  return generator, discriminator
