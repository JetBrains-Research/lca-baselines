  import bcc

# Define the BPF programs
count_program = bcc.BPFProgram("""
    int count_strlen(struct pt_regs *ctx) {
        return 1;
    }
""")

resolver_program = bcc.BPFProgram("""
    int resolver_strlen(struct pt_regs *ctx) {
        return 0;
    }
""")

# Define the functions to get the symbol of the indirect function, set the addresses of the resolver and implementation functions, and find the offset of the implementation function
def get_indirect_function_symbol():
    return bcc.get_user_functions_and_addresses("strlen")[0]

def set_resolver_function_address(address):
    resolver_program.set_function_address("resolver_strlen", address)

def find_implementation_function_offset(symbol):
    return bcc.find_library("libc.so.6").get_function_offset(symbol)

# Define the main function
def main():
    # Get the symbol of the indirect function
    symbol = get_indirect_function_symbol()

    # Find the offset of the implementation function
    offset = find_implementation_function_offset(symbol)

    # Set the addresses of the resolver and implementation functions
    set_resolver_function_address(offset)

    # Attach the counting BPF program to the implementation function
    count_program.attach_kprobe(event="resolver_strlen", fn_name="count_strlen")

    # Enter a loop where we sleep for one second, print the histogram, and clear the histogram
    while True:
        try:
            bcc.sleep(1)
            print(count_program.get_table("counts").items_lookup_and_delete_batch())
        except KeyboardInterrupt:
            break

if __name__ == "__main__":
    main()