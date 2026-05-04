from Framework.Event import StreamID
from deviceCount import deviceCount


fn chooseDevice(id: StreamID) -> Int:
    return Int(id) % deviceCount()
