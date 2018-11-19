import os
import sys
import multiprocessing
import errno
import signal
if os.name == 'nt':
    import msvcrt
else:
    import termios
from utils.log import log_e, log_i

def wait_key(tip="Press any key to continue ..."):
    ''' Wait for a key press on the console and return it. '''
    if tip is not None:
        print(tip)
    result = None
    if os.name == 'nt':
        result = msvcrt.getch()
    else:
        fd = sys.stdin.fileno()

        oldterm = termios.tcgetattr(fd)
        newattr = termios.tcgetattr(fd)
        newattr[3] = newattr[3] & ~termios.ICANON & ~termios.ECHO
        termios.tcsetattr(fd, termios.TCSANOW, newattr)

        try:
            result = sys.stdin.read(1)
        except IOError:
            pass
        finally:
            termios.tcsetattr(fd, termios.TCSAFLUSH, oldterm)
    print(result, end="")

    return result

def is_main_process():
    return type(multiprocessing.current_process()) != multiprocessing.Process

def ask(question, posstr="y", negstr="n", ansretry=True, ansdefault=None, timeout_sec=None):
    def timeout_interrupt(signum, frame):
        raise TimeoutError()

    assert not timeout_sec or ansdefault is not None, "Default answer need to be set when timeout is enabled"

    if ansretry is False:
        ansretry = 1
    elif ansretry is True:
        ansretry = float('inf')
    else:
        assert isinstance(ansretry, int)

    posstr = posstr.lower()
    negstr = negstr.lower()
    if ansdefault is not None:
        assert isinstance(ansdefault, bool)
        if ansdefault:
            ansdefault_str = posstr.lower()
            posstr = posstr.upper()
        else:
            ansdefault_str = negstr.lower()
            negstr = negstr.upper()
    else:
        assert ansretry == float('inf'), "No default answer for retry fallback"

    retry_count = 0
    while True:
        if timeout_sec:
            alarm_handler_ori = signal.signal(signal.SIGALRM, timeout_interrupt)
            signal.alarm(timeout_sec)
        try:
            ans = input(question + (" (timeout={}s)".format(timeout_sec) if timeout_sec else "") + " ({}|{}):".format(posstr, negstr))
        except TimeoutError:
            print()
            log_i("Answer timeout! using default answer: " + ansdefault_str)
            return ansdefault
        finally:
            if timeout_sec:
                signal.alarm(0)
                signal.signal(signal.SIGALRM, alarm_handler_ori)

        retry_count += 1

        if ans.lower() == posstr.lower():
            return True
        elif ans.lower() == negstr.lower():
            return False
        elif ansdefault is not None and not ans:
            return ansdefault
        else:
            if retry_count < ansretry:
                log_e("Illegal answer! Retry")
                continue
            else:
                # not possible to reach here when ansdefault is None
                log_e("Illegal answer! Using default answer: " + ansdefault_str)
                return ansdefault

def mkdir_p(dir_path):
    try:
        os.makedirs(dir_path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

