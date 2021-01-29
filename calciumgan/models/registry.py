from calciumgan.models import utils

_MODELS = dict()


def register(name):

  def add_to_dict(fn):
    global _MODELS
    _MODELS[name] = fn
    return fn

  return add_to_dict


def get_models(hparams, summary):
  if hparams.model not in _MODELS:
    print('model {} not found'.format(hparams.model))
    exit()

  generator, discriminator = _MODELS[hparams.model](hparams)

  summary.scalar('model/trainable_parameters/generator',
                 utils.count_trainable_params(generator))
  summary.scalar('model/trainable_parameters/discriminator',
                 utils.count_trainable_params(discriminator))

  gen_summary = utils.model_summary(hparams, generator)
  disc_summary = utils.model_summary(hparams, discriminator)

  if hparams.verbose == 2:
    print(gen_summary)
    print('')
    print(disc_summary)

  return generator, discriminator
