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

  G, D = _MODELS[hparams.model](hparams)

  summary.scalar('model/trainable_parameters/generator',
                 utils.count_trainable_params(G))
  summary.scalar('model/trainable_parameters/discriminator',
                 utils.count_trainable_params(D))

  G_summary = utils.model_summary(hparams, G)
  D_summary = utils.model_summary(hparams, D)

  if hparams.verbose == 2:
    print(G_summary)
    print('')
    print(D_summary)

  return G, D
