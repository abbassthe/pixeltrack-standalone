from MojoSerial.Framework.ESPluginFactory import Registry as ESRegistry
from MojoSerial.Framework.PluginFactory import (
    fwkModule,
    Registry as EDRegistry,
)

from MojoSerial.plugin_PixelVertexFinding.PixelVertexProducerCUDA import (
    PixelVertexProducerCUDA,
)


def init(
    mut esreg: ESRegistry,
    mut edreg: EDRegistry,
):
    fwkModule[PixelVertexProducerCUDA](edreg)
