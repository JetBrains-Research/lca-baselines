 Here is a Python script that uses the bcc library to create a histogram of system-wide strlen return values. The script assumes that strlen is an indirect function.

```python
import bcc
import sys
import time

class StrlenHistogram(bcc.BCCProgram):
    def __init__(self):
        super(StrlenHistogram, self).__init__("strlen_histogram")
        self.hist = {}

    def syscall_trace(self, pid, comm, syscall_num):
        if syscall_num == 252:  # sys_write
            self.set_bpf_local("resolver", self.args[0])
            self.set_bpf_local("implementation", self.args[1])

    def strlen_count(self, resolver, implementation, offset):
        self.set_bpf_local("resolver", resolver)
        self.set_bpf_local("implementation", implementation)
        self.set_bpf_local("offset", offset)
        self.attach_kprobe(event="strlen", fn_name="__kernel_vsyscall")

    def find_implementation_offset(self, resolver):
        self.set_bpf_local("resolver", resolver)
        self.attach_kprobe(event="syscall", fn_name="sys_call_table_get")
        self.start()
        sys_call_table = self.get_field(event="syscall", data_struct="sys_call_table")[0]
        self.stop()
        for i in range(len(sys_call_table)):
            sys_call = sys_call_table[i]
            if sys_call[0] == self.resolver:
                return sys_call[1]
        return -1

def get_indirect_function_symbol(name):
    # This function is not part of bcc library, you may need to implement it yourself
    # ...
    pass

def main():
    strlen_hist = StrlenHistogram()
    resolver_symbol = get_indirect_function_symbol("strlen")
    implementation_offset = strlen_hist.find_implementation_offset(resolver_symbol)
    if implementation_offset != -1:
        strlen_hist.strlen_count(resolver_symbol, b"strlen", implementation_offset)
        while True:
            time.sleep(1)
            print(strlen_hist.hist)
            strlen_hist.hist.clear()
            try:
                strlen_hist.detach()
            except Exception:
                pass
            try:
                strlen_hist.attach(strlen_hist.strlen_count, resolver_symbol, b"strlen", implementation_offset)
            except Exception:
                pass
            try:
                strlen_hist.start()
            except Exception:
                pass
            except KeyboardInterrupt:
                strlen_hist.stop()
                break

if __name__ == "__main__":
    main()
```

Please note that the `get_indirect_function_symbol` function is not part of the bcc library and you may need to implement it yourself based on your specific system and architecture. Also, this script assumes that the sys_call_table is a global variable and that the sys_call_table_get function returns the sys_call_table as a single element array. You may need to adjust these assumptions based on your system.

Lastly, this script does not handle the case where the strlen function is not an indirect function or is not present in the system. You may want to add error handling for such cases.