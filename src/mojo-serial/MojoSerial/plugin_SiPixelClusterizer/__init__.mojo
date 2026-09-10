from MojoSerial.Framework.ESPluginFactory import (
    fwkEventSetupModule,
    Registry as ESRegistry,
)
from MojoSerial.Framework.PluginFactory import (
    fwkModule,
    Registry as EDRegistry,
)
from MojoSerial.plugin_SiPixelClusterizer.SiPixelFedCablingMapGPUWrapperESProducer import (
    SiPixelFedCablingMapGPUWrapperESProducer,
)
from MojoSerial.plugin_SiPixelClusterizer.SiPixelGainCalibrationForHLTGPUESProducer import (
    SiPixelGainCalibrationForHLTGPUESProducer,
)
from MojoSerial.plugin_SiPixelClusterizer.SiPixelRawToClusterCUDA import (
    SiPixelRawToClusterCUDA,
)


def init(
    mut esreg: ESRegistry,
    mut edreg: EDRegistry,
):
    fwkEventSetupModule[SiPixelFedCablingMapGPUWrapperESProducer](esreg)
    fwkEventSetupModule[SiPixelGainCalibrationForHLTGPUESProducer](esreg)
    fwkModule[SiPixelRawToClusterCUDA](edreg)
