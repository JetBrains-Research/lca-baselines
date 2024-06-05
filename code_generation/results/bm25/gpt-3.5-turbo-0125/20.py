```python
import basilisk
from basilisk import unitTestSim as sim
from basilisk import unitTestSupport as support
from basilisk import spice_interface as spice

class MySimulation:
    def __init__(self):
        self.scSim = sim.SimBaseClass()
        self.scSim.TotalSim.logThisMessage('Hubble States:')
    
    def accessSpiceKernel(self):
        spice.loadSpiceKernel('path_to_kernel')
        hubbleState = spice.getHubbleState()
        print(hubbleState)
    
def main():
    for i in range(12):
        mySim = MySimulation()
        mySim.accessSpiceKernel()
    
    # class Controller:
    #     def __init__(self):
    #         spice.loadSpiceKernel('path_to_kernel')
    #         spice.unloadSpiceKernels()

if __name__ == "__main__":
    main()
```