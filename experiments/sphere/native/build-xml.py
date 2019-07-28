import openmc

# Instantiate planar surfaces
x1 = openmc.XPlane(surface_id=1, x0=-10.5)

x6 = openmc.XPlane(surface_id=6, x0=10.5)
y1 = openmc.YPlane(surface_id=11, y0=-10.5)

y6 = openmc.YPlane(surface_id=16, y0=10.5)
z1 = openmc.ZPlane(surface_id=21, z0=-10.5)

z6 = openmc.ZPlane(surface_id=26, z0=10.5)

sphere = openmc.Sphere(r=4.)

# Set vacuum boundary conditions on outside
for surface in [x1, x6, y1, y6, z1, z6]:
    surface.boundary_type = 'vacuum'

# Instantiate Cells
inner_sphere = openmc.Cell(cell_id=1, name="inner sphere")
outer_box = openmc.Cell(cell_id=3, name='outer box')

outer_cube = +x1 & -x6 & +y1 & -y6 & +z1 & -z6

# Use surface half-spaces to define regions
# middle_box.region = middle_cube & outside_inner_cube
outer_box.region = outer_cube & +sphere

# Register Materials with Cells
materials = openmc.Materials.from_xml("materials.xml")
inner_sphere.fill = materials[0]

# Instantiate root universe
root = openmc.Universe(universe_id=0, name='root universe')
root.add_cells([inner_sphere, outer_box])

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
