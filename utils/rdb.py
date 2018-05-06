from __future__ import print_function
import pdb
import socket
import sys
class Rdb(pdb.Pdb):
    def __init__(self, port=0):
        self.old_stdout = sys.stdout
        self.old_stdin = sys.stdin
        self.skt = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.skt.bind(('0.0.0.0', port))
        binded_port = self.skt.getsockname()[1]
        print("(Rdb) Listenning on 0.0.0.0:%d" % (binded_port,))
        self.skt.listen(1)
        (clientsocket, address) = self.skt.accept()
        print("(Rdb) Accepted on 0.0.0.0:%d from %s:%d" % ((binded_port, ) + tuple(clientsocket.getsockname()[:2])))
        handle = clientsocket.makefile('rw')
        print("(Rdb) Connected", file=handle)
        pdb.Pdb.__init__(self, completekey='tab', stdin=handle, stdout=handle)
        sys.stdout = sys.stdin = handle

    def do_continue(self, arg):
        sys.stdout = self.old_stdout
        sys.stdin = self.old_stdin
        self.skt.close()
        self.set_continue()
        return 1
    do_c = do_cont = do_continue

def set_trace():
    Rdb().set_trace(sys._getframe().f_back)
