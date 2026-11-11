"""Temporary Windows sleep inhibition for long experiments; OpenAI Codex assisted."""
from contextlib import contextmanager
import os


@contextmanager
def keep_awake():
    """Keep the system running, without changing its screen or persistent power plan."""
    previous = None
    if os.name == 'nt':
        import ctypes
        set_state = ctypes.windll.kernel32.SetThreadExecutionState
        set_state.argtypes = [ctypes.c_uint]
        set_state.restype = ctypes.c_uint
        previous = set_state(0x80000001)  # ES_CONTINUOUS | ES_SYSTEM_REQUIRED
        if not previous:
            raise OSError('Could not inhibit automatic sleep during the experiment')
    try:
        yield
    finally:
        if previous is not None:
            set_state(previous)


def wait_for_process(pid):
    """Hold the temporary request until the specified experiment process exits."""
    if os.name != 'nt':
        raise RuntimeError('This helper is for Windows')
    import ctypes
    kernel = ctypes.windll.kernel32
    kernel.OpenProcess.argtypes = [ctypes.c_uint, ctypes.c_int, ctypes.c_uint]
    kernel.OpenProcess.restype = ctypes.c_void_p
    kernel.WaitForSingleObject.argtypes = [ctypes.c_void_p, ctypes.c_uint]
    kernel.WaitForSingleObject.restype = ctypes.c_uint
    kernel.CloseHandle.argtypes = [ctypes.c_void_p]
    handle = kernel.OpenProcess(0x00100000, 0, pid)  # SYNCHRONIZE
    if not handle:
        raise OSError(f'Could not monitor experiment process {pid}')
    try:
        with keep_awake():
            print(f'Keeping system awake while experiment PID {pid} runs', flush=True)
            result = kernel.WaitForSingleObject(handle, 0xFFFFFFFF)
            if result != 0:
                raise OSError('Waiting for experiment failed')
    finally:
        kernel.CloseHandle(handle)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pid', type=int, required=True)
    wait_for_process(parser.parse_args().pid)
