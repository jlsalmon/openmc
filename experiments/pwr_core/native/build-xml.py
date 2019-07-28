import openmc

model = openmc.examples.pwr_core()
model.export_to_xml(".")