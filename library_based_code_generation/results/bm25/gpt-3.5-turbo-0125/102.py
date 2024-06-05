import seedemu

def create_stub_as(asn, exchange):
    asys = seedemu.createAutonomousSystem(asn)
    router = seedemu.createRouter(f"Router_{asn}")
    host1 = seedemu.createHostsOnNetwork(f"Host1_{asn}", "Network1")
    host2 = seedemu.createHostsOnNetwork(f"Host2_{asn}", "Network1")
    seedemu.joinNetwork(host1, "Network1")
    seedemu.joinNetwork(host2, "Network1")
    seedemu.joinInternetExchange(router, exchange)
    seedemu.setAutonomousSystem(asn, asys)
    return asys

seedemu.makeEmulatorBaseWith5StubASAndHosts()
seedemu.makeEmulatorBaseWith10StubASAndHosts()

as1 = create_stub_as(100, "Exchange1")
as2 = create_stub_as(200, "Exchange2")
as3 = create_stub_as(300, "Exchange3")

seedemu.shouldMerge(as1, as2)
seedemu.shouldMerge(as2, as3)
seedemu.shouldMerge(as1, as3)

seedemu.dumpEmulatorState("emulator_state.bin")