from seed_emulator import *

emulation = makeEmulatorBaseWith5StubASAndHosts()

as150 = emulation.getAsIsds(150)
as151 = emulation.getAsIsds(151)
as152 = emulation.getAsIsds(152)

web_service = WebService("web")
router = createHost("router0")

web_service.installService("web")

net0 = Network("net0")
net0.join(router)
net0.join(web_service)

as150.join(net0)
as151.join(net0)
as152.join(net0)

as150.markAsEdge()
as152.markAsEdge()

as150_provider = makeTransitAs(150)
as152_provider = makeStubAsWithHosts(152)

as150_provider.configureAsEvpnProvider(as152_provider)

internet_exchange = InternetExchange(100)
internet_exchange.join(as150)
internet_exchange.join(as151)

internet_exchange.join(as150_provider)

as150_provider.join(as152)

gen_emulation_files(emulation, "./cross-connect")
createEmulation(emulation, "Docker", "self-managed network")