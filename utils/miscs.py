import os
import sys
import multiprocessing
import errno
if os.name == 'nt':
    import msvcrt
else:
    import termios
from utils.log import log_e

def wait_key(tip=False):
    ''' Wait for a key press on the console and return it. '''
    if tip:
        print("Press any key to continue ...")
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

    return result

def is_main_process():
    return type(multiprocessing.current_process()) != multiprocessing.Process

def ask(question, posstr="y", negstr="n", ansretry=True, ansdefault=None):
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
            posstr = posstr.upper()
        else:
            negstr = negstr.upper()
    else:
        assert ansretry == float('inf'), "No default answer for retry fallback"

    retry_count = 0
    while True:
        ans = input(question + " ({}|{}):".format(posstr, negstr))
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
                log_e("Illegal answer! Using default answer: " + negstr.lower())
                return ansdefault

def mkdir_p(dir_path):
    try:
        os.makedirs(dir_path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise
