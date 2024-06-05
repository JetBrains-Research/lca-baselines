```python
from bcc import BPF

bpf_text = """
#include <uapi/linux/ptrace.h>

BPF_HASH(strlen_hist, u64);

int count_strlen_ret(struct pt_regs *ctx) {
    u64 key = PT_REGS_RC(ctx);
    u64 *val = strlen_hist.lookup_or_init(&key, 0);
    (*val)++;
    return 0;
}
"""

b = BPF(text=bpf_text)
b.attach_kprobe(event_re="^strlen$", fn_name="count_strlen_ret")

while True:
    try:
        sleep(1)
        print("Histogram:")
        for k, v in b["strlen_hist"].items():
            print("Length: %d Count: %d" % (k.value, v.value))
        b["strlen_hist"].clear()
    except KeyboardInterrupt:
        break
```