_ALGORITHMS = dict()


def register(name):

  def add_to_dict(fn):
    global _ALGORITHMS
    _ALGORITHMS[name] = fn
    return fn

  return add_to_dict


def get_algorithm(hparams, generator, discriminator):
  if hparams.algorithm not in _ALGORITHMS:
    print('Algorithm {} not found'.format(hparams.algorithm))
    exit()
  return _ALGORITHMS[hparams.algorithm](hparams, generator, discriminator)
