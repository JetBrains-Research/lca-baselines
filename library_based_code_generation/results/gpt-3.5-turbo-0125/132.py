```python
from bcc import BPF

bpf_text = """
#include <uapi/linux/ptrace.h>

BPF_HASH(counts, u64);

int count_strlen(struct pt_regs *ctx) {
    u64 key = (u64)PT_REGS_RC(ctx);
    u64 *val = counts.lookup_or_init(&key, 0);
    (*val)++;
    return 0;
}
"""

b = BPF(text=bpf_text)
b.attach_kprobe(event_re="^strlen$", fn_name="count_strlen")

while True:
    try:
        sleep(1)
        print("Histogram:")
        for k, v in b["counts"].items():
            print("strlen return value: %d, count: %d" % (k, v))
        b["counts"].clear()
    except KeyboardInterrupt:
        break
```