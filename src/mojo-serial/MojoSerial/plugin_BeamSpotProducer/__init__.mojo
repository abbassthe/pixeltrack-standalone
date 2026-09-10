from MojoSerial.Framework.ESPluginFactory import (
    fwkEventSetupModule,
    Registry as ESRegistry,
)
from MojoSerial.Framework.PluginFactory import (
    fwkModule,
    Registry as EDRegistry,
)

from MojoSerial.plugin_BeamSpotProducer.BeamSpotESProducer import (
    BeamSpotESProducer,
)
from MojoSerial.plugin_BeamSpotProducer.BeamSpotToPOD import (
    BeamSpotToPOD,
)


def init(
    mut esreg: ESRegistry,
    mut edreg: EDRegistry,
):
    fwkEventSetupModule[BeamSpotESProducer](esreg)
    fwkModule[BeamSpotToPOD](edreg)
