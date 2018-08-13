import os
import sys
import multiprocessing
import torch

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

def load_pretrained_loose(model, pretrained_state_dict, pause_model_mismatch=True, confirm_model_size_mismatch=True):
    from collections import OrderedDict
    state_dict = model.state_dict()
    model_missing_keys = set(list(pretrained_state_dict.keys())) - set(list(state_dict.keys()))
    model_extra_keys = set(list(state_dict.keys())) - set(list(pretrained_state_dict.keys()))
    if len(model_missing_keys) > 0:
        print("[Warning] Model missing keys: " + str(model_missing_keys))
    if len(model_extra_keys) > 0:
        print("[Warning] Model extra keys: " + str(model_extra_keys))
    if pause_model_mismatch and (len(model_missing_keys) > 0 or len(model_extra_keys) > 0):
        import utils.miscs as miscs
        print("Press any key to continue")
        miscs.wait_key()

    for k, v in pretrained_state_dict.items():
        if k in model_missing_keys:
            continue
        model_k_size = state_dict[k].size()
        pretr_k_size = v.size()
        if model_k_size != pretr_k_size and k.endswith(".weight") and len(model_k_size) == len(pretr_k_size) and len(model_k_size) == 4:
            assert model_k_size[0] >= pretr_k_size[0] and model_k_size[1] >= pretr_k_size[1]
            # Output more than pretrained
            if confirm_model_size_mismatch and model_k_size[0] > pretr_k_size[0]:
                print("[Warning] Model output channel size({}) larger than pretrained({}) for {}".format(model_k_size[0], pretr_k_size[0], k))
                if not ask("Would you like adaptive loading ?"):
                    print("[Exit] No adaptive loading")
                    sys.exit(1)

            # Input more than pretrained
            if confirm_model_size_mismatch and model_k_size[1] > pretr_k_size[1]:
                print("[Warning] Model input channel size({}) larger than pretrained({}) for {}.".format(model_k_size[1], pretr_k_size[1], k))
                if not ask("Would you like adaptive loading ?"):
                    print("[Exit] No adaptive loading")
                    sys.exit(1)

            state_dict[k][:pretr_k_size[0], :pretr_k_size[1]] = v[:, :]
        else:
            state_dict[k] = v
    model.load_state_dict(state_dict)
