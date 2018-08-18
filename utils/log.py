from termcolor import colored, cprint

__all__ = ["log_text", "log_level", "log_i", "log_w", "log_e", "log_q", "log_suc", "log_fail", "log_progress"]

def log_text(text, **kwargs):
    print(text, **kwargs)

def log_level(level, text, **kwargs):
    print(colored("[" + level + "]", "white", "on_red") + " " + text, **kwargs)

def log_i(text, **kwargs):
    log_level("INFO", text, **kwargs)

def log_w(text, **kwargs):
    log_level("WARNING", text, **kwargs)

def log_e(text, **kwargs):
    log_level("ERROR", text, **kwargs)

def log_q(text, **kwargs):
    log_level("EXIT", text, **kwargs)

def log_suc(text, **kwargs):
    log_level("SUCCESS", text, **kwargs)

def log_fail(text, **kwargs):
    log_level("FAILED", text, **kwargs)

def log_progress(text):
    cprint("==> " + text, "green")
