#!/usr/bin/env python
import os
import vtk
import argparse
import meshio


def convertFile(infile, outfile):
    if os.path.isfile(infile):
        basename = os.path.basename(infile)
        print("Copying file:", basename)
        basename = os.path.splitext(basename)[0]
        outfile = os.path.join(basename + ".stl")
        mesh = meshio.read(infile)
        del mesh.cells["line"]
        meshio.write(outfile, mesh)
        # reader = vtk.vtkGenericDataObjectReader()
        # reader.SetFileName(infile)
        # reader.Update()
        # writer = vtk.vtkSTLWriter()
        # writer.SetInputConnection(reader.GetOutputPort())
        # writer.SetFileName(outfile)
        # return writer.Write() == 1
    return False


def convertFiles(infile, outfile):
    convertFile(infile, outfile)


def run(args):
    convertFiles(args.infile, args.outfile)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="VTK to STL converter")
    parser.add_argument('infile', help="Path to input file.")
    parser.add_argument('--outfile', '-o', default='output', help="Path to output file.")
    parser.set_defaults(func=run)
    args = parser.parse_args()
    ret = args.func(args)
