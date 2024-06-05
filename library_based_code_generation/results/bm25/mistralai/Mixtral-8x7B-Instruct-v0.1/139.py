 import pydpf

pydpf.should_average = False
pydpf.should_start_server = False

result = pydpf.__generate_code.download\_multi\_stage\_cyclic\_result(
"https://drive.google.com/file/d/1wvWc3T-JXb\_-YXBZzYzBZzYzBZzYzBZz/view?usp=share\_link")

model = pydpf.create\_model\_from\_result(result)
model.display\_state()

result\_info = model.result\_info()
print("Number of results:", result\_info.get\_number\_of\_results())

cyclic\_support = model.cyclic\_support()
print("Number of stages:", cyclic\_support.get\_num\_stages())
print("Number of sectors in each stage:", [len(stage.get\_sectors\_scoping()) for stage in cyclic\_support.get\_stages()])
print("Number of nodes in the first stage's base sector:", cyclic\_support.get\_sectors\_scoping(0)[0].get\_nodes\_count())

expanded\_displacements = pydpf.cyclic\_expanded\_displacement(model, sectors\_to\_expand\_first\_stage=[0], sectors\_to\_expand\_stage\_by\_stage=[[1]])
total\_deformation = expanded\_displacements.get\_total\_deformation()
expanded\_mesh = expanded\_displacements.get\_expanded\_mesh()

pydpf.create\_mesh\_and\_field\_mapped(expanded\_mesh, total\_deformation)

# Plotting and checking results
import pyvista as pv

plotter = pv.Plotter()
plotter.add\_mesh(expanded\_mesh, scalars=total\_deformation, show\_scalar\_bar=False)
plotter.show()

harmonic\_index = model.result\_definition().get\_time\_frequency\_support().get\_harmonic\_index()
print("Time frequency support (harmonic index):", harmonic\_index)

displacements = model.result\_definition().get\_displacements()
for node in model.mesh().nodes():
node\_displacements = [displacements[i][node.get\_index()] for i in range(len(displacements))]
if node\_displacements != node\_displacements[0]:
raise ValueError("Displacement values are not the same on all nodes.")

print("Displacement values are the same on all nodes.")