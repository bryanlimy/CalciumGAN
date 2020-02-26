from .utils import count_trainable_params

_GENERATORS, _DISCRIMINATORS = dict(), dict()


def generator_register(fn):
  global _GENERATORS
  _GENERATORS[fn.__name__] = fn
  return fn


def discriminator_register(fn):
  global _DISCRIMINATORS
  _DISCRIMINATORS[fn.__name__] = fn
  return fn


def get_models(hparams, summary):
  if hparams.generator not in _GENERATORS:
    print('generator model {} not found'.format(hparams.generator))
    exit()
  if hparams.discriminator not in _DISCRIMINATORS:
    print('discriminator model {} not found'.format(hparams.discriminator))
    exit()
  generator = _GENERATORS[hparams.generator](hparams)
  discriminator = _DISCRIMINATORS[hparams.discriminator](hparams)

  summary.scalar('model/trainable_parameters/generator',
                 count_trainable_params(generator))
  summary.scalar('model/trainable_parameters/discriminator',
                 count_trainable_params(discriminator))
  return generator, discriminator
