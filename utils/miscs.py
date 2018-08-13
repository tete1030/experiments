import os
import sys
import multiprocessing
import errno

if os.name == 'nt':
    import msvcrt
else:
    import termios

def wait_key():
    ''' Wait for a key press on the console and return it. '''
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

def ask(question, posstr="y", negstr="n", ansretry=False, ansdefault=False):
    if ansretry is False:
        ansretry = 1
    else:
        assert isinstance(ansretry, int)

    retry_count = 0
    while True:
        ans = input(question + " (%s|%s) :" % (posstr, negstr))
        retry_count += 1

        if ans == posstr:
            return True
        elif ans == negstr:
            return False
        else:
            if retry_count < ansretry:
                print("[Error] Illegal answer! Retry")
                continue
            else:
                print("[Error] Illegal answer!")
                return ansdefault

def mkdir_p(dir_path):
    try:
        os.makedirs(dir_path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise
