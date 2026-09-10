from MojoSerial.Framework.ESPluginFactory import (
    fwkEventSetupModule,
    Registry as ESRegistry,
)
from MojoSerial.Framework.PluginFactory import (
    fwkModule,
    Registry as EDRegistry,
)

from MojoSerial.plugin_Validation.CountValidator import (
    CountValidator,
)


def init(
    mut esreg: ESRegistry,
    mut edreg: EDRegistry,
):
    fwkModule[CountValidator](edreg)