import basilisk
from basilisk.sim import simple_sim
from basilisk.sim import spice_interface

class MySimulation:
    def __init__(self):
        self.scSim = simple_sim.SimpleSim()
        self.scSim.addSpacecraft("spacecraft")
        self.scSim["spacecraft"].addState(spice_interface.SpiceState())
    
    def accessSpiceKernel(self):
        spice_interface.loadSpiceKernel()
        state = spice_interface.getSpiceState("Hubble")
        print(state)
    
class Controller:
    # def __init__(self):
    #     spice_interface.loadSpiceKernel()

def main():
    for i in range(12):
        sim = MySimulation()
        sim.accessSpiceKernel()
    
if __name__ == "__main__":
    main()