import openmc

# Instantiate planar surfaces
x1 = openmc.XPlane(surface_id=1, x0=-10)
x3 = openmc.XPlane(surface_id=3, x0=-5)
x4 = openmc.XPlane(surface_id=4, x0=5)
x6 = openmc.XPlane(surface_id=6, x0=10)
y1 = openmc.YPlane(surface_id=11, y0=-10)
y3 = openmc.YPlane(surface_id=13, y0=-5)
y4 = openmc.YPlane(surface_id=14, y0=5)
y6 = openmc.YPlane(surface_id=16, y0=10)
z1 = openmc.ZPlane(surface_id=21, z0=-10)
z3 = openmc.ZPlane(surface_id=23, z0=-5)
z4 = openmc.ZPlane(surface_id=24, z0=5)
z6 = openmc.ZPlane(surface_id=26, z0=10)

# Set vacuum boundary conditions on outside
for surface in [x1, x6, y1, y6, z1, z6]:
    surface.boundary_type = 'vacuum'

# Instantiate Cells
inner_box = openmc.Cell(cell_id=1, name='inner box')
middle_box = openmc.Cell(cell_id=2, name='middle box')
outer_box = openmc.Cell(cell_id=3, name='outer box')

# Use each set of six planes to create solid cube regions. We can then use these
# to create cubic shells.
inner_cube = +x3 & -x4 & +y3 & -y4 & +z3 & -z4
# middle_cube = +x2 & -x5 & +y2 & -y5 & +z2 & -z5
outer_cube = +x1 & -x6 & +y1 & -y6 & +z1 & -z6
outside_inner_cube = -x3 | +x4 | -y3 | +y4 | -z3 | +z4

# Use surface half-spaces to define regions
# middle_box.region = middle_cube & outside_inner_cube
inner_box.region = inner_cube
outer_box.region = outer_cube & ~inner_cube

# Register Materials with Cells
materials = openmc.Materials.from_xml("materials.xml")
inner_box.fill = materials[0]

# Instantiate root universe
root = openmc.Universe(universe_id=0, name='root universe')
root.add_cells([inner_box, outer_box])

# Instantiate a Geometry, register the root Universe, and export to XML
geometry = openmc.Geometry(root)
geometry.export_to_xml("geometry.xml")


###############################################################################
#                   Exporting to OpenMC plots.xml File
###############################################################################

plot = openmc.Plot(plot_id=1)
plot.origin = [0, 0, 0]
plot.width = [25, 25]
plot.pixels = [200, 200]
plot.color_by = 'cell'

# Instantiate a Plots collection and export to XML
plot_file = openmc.Plots([plot])
plot_file.export_to_xml("plots.xml")
