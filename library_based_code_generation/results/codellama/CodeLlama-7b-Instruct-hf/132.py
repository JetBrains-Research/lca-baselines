  import bcc
import ctypes
import time

# Define the BPF programs
count_bpf = bcc.BPF(text="""
#include <uapi/linux/ptrace.h>
#include <linux/blkdev.h>

BPF_HISTOGRAM(dist, int);
""")

# Define the functions to get the symbol of the indirect function, set the addresses of the resolver and implementation functions, and find the offset of the implementation function
def get_symbol(name):
    return ctypes.CDLL("libc.so.6").dlsym(ctypes.c_void_p(0), name.encode())

def set_resolver(resolver):
    ctypes.CDLL("libc.so.6").dlsym(ctypes.c_void_p(0), b"__resolver".encode()) = resolver

def find_offset(resolver, impl):
    return ctypes.addressof(resolver) - ctypes.addressof(impl)

# Define the main function
def main():
    # Get the symbol of the indirect function
    indirect_func = get_symbol("strlen")

    # Set the addresses of the resolver and implementation functions
    set_resolver(indirect_func)

    # Find the offset of the implementation function
    offset = find_offset(indirect_func, indirect_func)

    # Attach the counting BPF program to the implementation function
    count_bpf.attach_kprobe(event="strlen", fn_name="dist")

    # Enter a loop where we sleep for one second, print the histogram, and clear the histogram
    while True:
        time.sleep(1)
        print(count_bpf.get_table("dist"))
        count_bpf.clear_table("dist")

# Run the main function
if __name__ == "__main__":
    main()
