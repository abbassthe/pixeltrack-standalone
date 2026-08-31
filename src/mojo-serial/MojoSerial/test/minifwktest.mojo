from std.pathlib import Path

from MojoSerial.plugin_SiPixelClusterizer.SiPixelFedCablingMapGPUWrapperESProducer import (
    SiPixelFedCablingMapGPUWrapperESProducer,
)
from MojoSerial.Framework.EventSetup import EventSetup
from MojoSerial.Framework.ESPluginFactory import ESPluginFactory
from MojoSerial.Framework.PluginFactory import PluginFactory
from MojoSerial.Framework.ESProducer import ESProducer
from MojoSerial.MojoBridge.DTypes import Typeable


def main() raises:
    var _esreg = MojoSerial.Framework.ESPluginFactory.Registry()
    var _edreg = MojoSerial.Framework.PluginFactory.Registry()
    MojoSerial.plugin_SiPixelClusterizer.init(_esreg, _edreg)
    var evt = EventSetup()

    for plugin in ESPluginFactory.getAll(_esreg):
        var esp = ESPluginFactory.create(plugin, "data", _esreg)
        esp.produce(evt)

    for plugin in PluginFactory.getAll(_edreg):
        print(plugin)

    # Lifetime registry extension
    _ = _esreg^
    _ = _edreg^
