_LOSSES = dict()


def losses_register(name):

  def add_to_dict(fn):
    global _LOSSES
    _LOSSES[name] = fn
    return fn

  return add_to_dict


def get_losses(hparams):
  if hparams.loss not in _LOSSES:
    print('Loss function {} not found'.format(hparams.loss))
    exit()
  generator_loss, discriminator_loss = _LOSSES[hparams.loss](hparams)
  return generator_loss, discriminator_loss
