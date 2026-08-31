from MojoSerial.Framework.PluginFactory import fwkModule

from MojoSerial.plugin_PixelVertexFinding.PixelVertexProducerCUDA import (
    PixelVertexProducerCUDA,
)


def init(
    mut esreg: MojoSerial.Framework.ESPluginFactory.Registry,
    mut edreg: MojoSerial.Framework.PluginFactory.Registry,
):
    fwkModule[PixelVertexProducerCUDA](edreg)
