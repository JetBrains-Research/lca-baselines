from seedemu import *

emulation = makeEmulatorBaseWith5StubASAndHosts()

AS150 = getNodeByAsnAndName(150, "AS150")
AS2 = getNodeByAsnAndName(2, "AS2")
AS151 = getNodeByAsnAndName(151, "AS151")

network1 = createNetwork(NetworkType.LAN, maskNetwork("10.0.0.0/24"))
network2 = createNetwork(NetworkType.LAN, maskNetwork("10.0.1.0/24"))
network3 = createNetwork(NetworkType.LAN, maskNetwork("10.0.2.0/24"))

addIxLink(AS150, AS2, mapIxAddress(100))
addIxLink(AS2, AS151, mapIxAddress(101))

bgp_attacker = BgpAttackerComponent()
bgp_attacker.hijackPrefix(AS151)
bgp_attacker.joinIX(AS2, mapIxAddress(100))

private_peering1 = joinNetwork(AS150, AS2, mapIxAddress(100))
private_peering2 = joinNetwork(AS151, AS2, mapIxAddress(101))
private_peering3 = joinNetwork(bgp_attacker, AS2, mapIxAddress(100))

__compileIxNetWorker(emulation)
__compileIxNetMaster(emulation)