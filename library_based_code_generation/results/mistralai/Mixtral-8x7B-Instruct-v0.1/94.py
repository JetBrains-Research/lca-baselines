 ```python
import os
import pyaedt

os.chdir("C:/temp")
edb = pyaedt.Hfss3dLayout()
edb_dir = os.path.join(os.getcwd(), "DCIR_Example_Board")
edb.newProject(Default=False, ProjectType="EDB", ProjectName="DCIR_Example_Board", Editable=True, EdbVersion="2022.1")
edb.setActiveDesign("DCIR_Example_Board")
edb.openHfssLayout()
edb.setWorkingDirectory(edb_dir)
edb.importLayout("DCIR_Example_Board.emp", "EDB", "C:/pyaedt/aedt_release/HFSS/AEDT/HFSS/Import_Exports/EDB_Layouts/DCIR_Example_Board.emp")

# Configure EDB for DCIR analysis
edb.configureDcirAnalysis()

# Create pin groups
pos_vrm_pins = ["VRM_P1_P", "VRM_P2_P"]
neg_vrm_pins = ["VRM_P1_N", "VRM_P2_N"]
pos_sink_pins = ["Sink_P1_P", "Sink_P2_P"]
neg_sink_pins = ["Sink_P1_N", "Sink_P2_N"]
edb.createPinGroups(pos_vrm_pins, "VRM_Positive")
edb.createPinGroups(neg_vrm_pins, "VRM_Negative")
edb.createPinGroups(pos_sink_pins, "Sink_Positive")
edb.createPinGroups(neg_sink_pins, "Sink_Negative")

# Create voltage and current sources
edb.createDCIRVoltageSource("VRM_Positive", "VRM_Negative", "VRM_VSource", "VRM_VSource", 1.8, "DC")
edb.createDCIRCurrentSource("Sink_Positive", "Sink_Negative", "Sink_ISource", "Sink_ISource", 1, "DC")

# Add SIwave DCIR analysis
analysis_name = "DCIR_Analysis"
edb.analyze(analysis_name, "DCIR", overwrite=True)

# Save and close EDB
edb.saveProject()
edb.closeHfssLayout()
edb.closeProject()

# Launch AEDT, import the configured EDB and analyze DCIR
aedt = pyaedt.Aedt(specified_version="2022.1")
aedt.newProject(project_name="DCIR_Example_Board", project_type="HFSS 3D Layout")
aedt.openProject(edb_dir)
aedt.analyzeDesign("DCIR_Example_Board", "DCIR_Analysis")

# Retrieve and print the loop resistance, current source, and via information from the DCIR element data
dcir_data = aedt.getDesignData("DCIR_Example_Board", "DCIR_Analysis")
loop_resistance = dcir_data["DCIR_Element_Data"]["LoopResistance"]
current_source = dcir_data["DCIR_Element_Data"]["CurrentSource"]
via_info = dcir_data["DCIR_Element_Data"]["ViaInfo"]
print(f"Loop Resistance: {loop_resistance}, Current Source: {current_source}, Via Info: {via_info}")

# Retrieve and print the voltage from the DCIR solution data
solution_data = aedt.getDesignData("DCIR_Example_Board", "DCIR_Analysis", "SolutionData")
voltage = solution_data["VRM_VSource"]
print(f"Voltage: {voltage}")

# Close the AEDT project and release the desktop
aedt.closeProject()
pyaedt.desktop.release()
```