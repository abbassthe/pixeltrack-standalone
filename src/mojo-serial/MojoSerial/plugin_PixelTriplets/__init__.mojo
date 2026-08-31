from MojoSerial.Framework.PluginFactory import fwkModule

from MojoSerial.plugin_PixelTriplets.CAHitNtupletCUDA import CAHitNtupletCUDA


def init(
    mut esreg: MojoSerial.Framework.ESPluginFactory.Registry,
    mut edreg: MojoSerial.Framework.PluginFactory.Registry,
):
    fwkModule[CAHitNtupletCUDA](edreg)
