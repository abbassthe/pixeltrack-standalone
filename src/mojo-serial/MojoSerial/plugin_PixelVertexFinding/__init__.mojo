from MojoSerial.Framework.PluginFactory import fwkModule

from MojoSerial.plugin_PixelVertexFinding.PixelVertexProducerCUDA import (
    PixelVertexProducerCUDA,
)


fn init(
    mut esreg: MojoSerial.Framework.ESPluginFactory.Registry,
    mut edreg: MojoSerial.Framework.PluginFactory.Registry,
):
    fwkModule[PixelVertexProducerCUDA](edreg)
