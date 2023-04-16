from .gan import Generator as GanGenerator, \
                 Discriminator as GanDiscriminator, \
                 trainer as GanTrainer

from .dcgan import Generator as DCGanGenerator, \
                   Discriminator as DCGanDiscriminator, \
                   trainer as DCGanTrainer              

from .cgan import Generator as CGanGenerator, \
                  Discriminator as CGanDiscriminator, \
                  trainer as CGanTrainer      