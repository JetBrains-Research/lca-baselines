 from bcc import BPF
from bcc.utils import printb
import ctypes
import time

prog = """
#include <uapi/linux/bpf.h>
#include <linux/pkt_cls.h>

struct data_t {
    __u32 len;
    __u64 addr;
};

BPF_PERF_OUTPUT(events);

int streq(void *data, int size) {
    struct data_t *ctx = data;
    bpf_perf_event_output(ctx, &events, BPF_F_CURRENT_CPU, ctx, sizeof(*ctx));
    return 0;
}

