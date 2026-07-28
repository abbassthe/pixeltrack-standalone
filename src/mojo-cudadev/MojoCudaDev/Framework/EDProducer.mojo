from MojoCudaDev.Framework.Event import Event
from MojoCudaDev.Framework.EventSetup import EventSetup
from MojoCudaDev.Framework.ProductRegistry import ProductRegistry


trait EDProducer(Defaultable):
    # this cannot raise
    fn __init__(out self, mut reg: ProductRegistry):
        ...

    fn produce(mut self, mut event: Event, ref eventSetup: EventSetup):
        ...

    fn endJob(mut self):
        ...
