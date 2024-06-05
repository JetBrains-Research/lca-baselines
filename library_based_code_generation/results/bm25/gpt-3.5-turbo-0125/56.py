import seed_emulator

emulator = seed_emulator.createEmulation()

AS150 = seed_emulator.makeStubAsWithHosts(emulator, "AS150", 4, 3)
AS151 = seed_emulator.makeStubAsWithHosts(emulator, "AS151", 1, 1)
AS152 = seed_emulator.makeStubAsWithHosts(emulator, "AS152", 1, 1)

seed_emulator.joinNetwork(AS150, "network1")
seed_emulator.joinNetwork(AS150, "network2")
seed_emulator.joinNetwork(AS150, "network3")

seed_emulator.joinNetwork(AS151, "network4")
seed_emulator.joinInternetExchange(AS151, "IX1")

seed_emulator.joinNetwork(AS152, "network5")
seed_emulator.joinInternetExchange(AS152, "IX1")

seed_emulator.shouldMerge(AS150, AS151)
seed_emulator.shouldMerge(AS150, AS152)

seed_emulator.gen_emulation_files(emulator, "emulation_state.bin")