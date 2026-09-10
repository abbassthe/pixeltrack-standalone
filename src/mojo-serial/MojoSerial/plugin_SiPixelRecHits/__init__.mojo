from MojoSerial.Framework.ESPluginFactory import (
    fwkEventSetupModule,
    Registry as ESRegistry,
)
from MojoSerial.Framework.PluginFactory import (
    fwkModule,
    Registry as EDRegistry,
)

from MojoSerial.plugin_SiPixelRecHits.PixelCPEFastESProducer import (
    PixelCPEFastESProducer,
)
from MojoSerial.plugin_SiPixelRecHits.SiPixelRecHitCUDA import (
    SiPixelRecHitCUDA,
)


def init(
    mut esreg: ESRegistry,
    mut edreg: EDRegistry,
):
    fwkEventSetupModule[PixelCPEFastESProducer](esreg)
    fwkModule[SiPixelRecHitCUDA](edreg)
