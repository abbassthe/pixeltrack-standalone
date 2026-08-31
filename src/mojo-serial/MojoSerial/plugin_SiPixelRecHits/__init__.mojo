from MojoSerial.Framework.ESPluginFactory import fwkEventSetupModule
from MojoSerial.Framework.PluginFactory import fwkModule

from MojoSerial.plugin_SiPixelRecHits.PixelCPEFastESProducer import (
    PixelCPEFastESProducer,
)
from MojoSerial.plugin_SiPixelRecHits.SiPixelRecHitCUDA import (
    SiPixelRecHitCUDA,
)


def init(
    mut esreg: MojoSerial.Framework.ESPluginFactory.Registry,
    mut edreg: MojoSerial.Framework.PluginFactory.Registry,
):
    fwkEventSetupModule[PixelCPEFastESProducer](esreg)
    fwkModule[SiPixelRecHitCUDA](edreg)
