from MojoSerial.Framework.ESPluginFactory import fwkEventSetupModule
from MojoSerial.Framework.PluginFactory import fwkModule

from MojoSerial.plugin_BeamSpotProducer.BeamSpotESProducer import (
    BeamSpotESProducer,
)
from MojoSerial.plugin_BeamSpotProducer.BeamSpotToPOD import (
    BeamSpotToPOD,
)


def init(
    mut esreg: MojoSerial.Framework.ESPluginFactory.Registry,
    mut edreg: MojoSerial.Framework.PluginFactory.Registry,
):
    fwkEventSetupModule[BeamSpotESProducer](esreg)
    fwkModule[BeamSpotToPOD](edreg)
