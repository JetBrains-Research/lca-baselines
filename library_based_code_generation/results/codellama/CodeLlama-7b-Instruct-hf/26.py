  from seedemu import Emulator

# Create an emulation environment
emulator = Emulator()

# Add layers to the environment
emulator.add_layer("Base")
emulator.add_layer("Routing")
emulator.add_layer("Ebgp")
emulator.add_layer("Ibgp")
emulator.add_layer("Ospf")
emulator.add_layer("WebService")

# Create Internet Exchanges with custom display names
emulator.add_ix("IX-1", "Custom Display Name 1")
emulator.add_ix("IX-2", "Custom Display Name 2")

# Create Transit Autonomous Systems and single-homed stub Autonomous Systems with various services
emulator.add_as("Transit AS", "Transit")
emulator.add_as("Stub AS", "Stub")
emulator.add_as("Service AS", "Service")

# Add a host with a customized IP address to one of the Autonomous Systems
emulator.add_host("Host 1", "10.0.0.1", "Transit AS")

# Create a real-world Autonomous System
emulator.add_as("Real-world AS", "Real-world")

# Enable remote access to one of the Autonomous System's network
emulator.add_remote_access("Transit AS", "10.0.0.2")

# Set up peering via a route server and private peering with different peer relationships
emulator.add_peering("Transit AS", "Stub AS", "Route Server")
emulator.add_peering("Transit AS", "Service AS", "Private")

# Add all the layers to the emulator
emulator.add_layer("Base")
emulator.add_layer("Routing")
emulator.add_layer("Ebgp")
emulator.add_layer("Ibgp")
emulator.add_layer("Ospf")
emulator.add_layer("WebService")

# Save the emulator to a component file
emulator.save("emulator.component")

# Render and compile the emulator
emulator.render()
emulator.compile()