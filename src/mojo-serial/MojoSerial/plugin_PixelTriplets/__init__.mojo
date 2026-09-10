from MojoSerial.Framework.ESPluginFactory import Registry as ESRegistry
from MojoSerial.Framework.PluginFactory import (
    fwkModule,
    Registry as EDRegistry,
)

from MojoSerial.plugin_PixelTriplets.CAHitNtupletCUDA import CAHitNtupletCUDA


def init(
    mut esreg: ESRegistry,
    mut edreg: EDRegistry,
):
    fwkModule[CAHitNtupletCUDA](edreg)
