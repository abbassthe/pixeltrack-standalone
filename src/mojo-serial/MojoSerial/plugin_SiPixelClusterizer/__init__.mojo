from MojoSerial.Framework.ESPluginFactory import fwkEventSetupModule
from MojoSerial.Framework.PluginFactory import fwkModule
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
    mut esreg: MojoSerial.Framework.ESPluginFactory.Registry,
    mut edreg: MojoSerial.Framework.PluginFactory.Registry,
):
    fwkEventSetupModule[SiPixelFedCablingMapGPUWrapperESProducer](esreg)
    fwkEventSetupModule[SiPixelGainCalibrationForHLTGPUESProducer](esreg)
    fwkModule[SiPixelRawToClusterCUDA](edreg)
