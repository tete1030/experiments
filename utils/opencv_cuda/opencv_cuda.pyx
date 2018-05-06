def get_device_count():
    return getCudaEnabledDeviceCount()

def set_device(idx):
    setDevice(idx)

def get_device():
    return getDevice()

def reset_device():
    resetDevice()