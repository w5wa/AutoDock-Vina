#! /usr/bin/env python
# -*- coding: utf-8 -*-
#
# AutoDock Vina – basic docking example
#
# Usage (all arguments are optional; defaults use the bundled example files):
#
#   python first_example.py \
#       --receptor 1iep_receptor.pdbqt \
#       --ligand   1iep_ligand.pdbqt \
#       --center_x 15.190 --center_y 53.903 --center_z 16.917 \
#       --size_x 20 --size_y 20 --size_z 20 \
#       --out      1iep_ligand_vina_out.pdbqt
#

import argparse
import os

from vina import Vina


def parse_args():
    parser = argparse.ArgumentParser(
        description='AutoDock Vina – basic docking example',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--receptor', default='1iep_receptor.pdbqt',
                        help='rigid receptor PDBQT file')
    parser.add_argument('--ligand', default='1iep_ligand.pdbqt',
                        help='ligand PDBQT file')
    parser.add_argument('--center_x', type=float, default=15.190,
                        help='X coordinate of the box center (Angstrom)')
    parser.add_argument('--center_y', type=float, default=53.903,
                        help='Y coordinate of the box center (Angstrom)')
    parser.add_argument('--center_z', type=float, default=16.917,
                        help='Z coordinate of the box center (Angstrom)')
    parser.add_argument('--size_x', type=float, default=20.0,
                        help='box size along X (Angstrom)')
    parser.add_argument('--size_y', type=float, default=20.0,
                        help='box size along Y (Angstrom)')
    parser.add_argument('--size_z', type=float, default=20.0,
                        help='box size along Z (Angstrom)')
    parser.add_argument('--exhaustiveness', type=int, default=32,
                        help='exhaustiveness of global search')
    parser.add_argument('--n_poses', type=int, default=20,
                        help='number of poses to generate')
    parser.add_argument('--out', default='1iep_ligand_vina_out.pdbqt',
                        help='output PDBQT file for docked poses')
    return parser.parse_args()


def main():
    args = parse_args()

    # Validate inputs
    for path, label in [(args.receptor, 'receptor'), (args.ligand, 'ligand')]:
        if not os.path.exists(path):
            raise FileNotFoundError('Cannot find %s file: %s' % (label, path))

    v = Vina(sf_name='vina')

    v.set_receptor(args.receptor)
    v.set_ligand_from_file(args.ligand)
    v.compute_vina_maps(
        center=[args.center_x, args.center_y, args.center_z],
        box_size=[args.size_x, args.size_y, args.size_z]
    )

    # Score the current pose
    energy = v.score()
    print('Score before minimization: %.3f (kcal/mol)' % energy[0])

    # Local minimization
    energy_minimized = v.optimize()
    print('Score after minimization : %.3f (kcal/mol)' % energy_minimized[0])
    minimized_out = os.path.splitext(args.out)[0] + '_minimized.pdbqt'
    v.write_pose(minimized_out, overwrite=True)
    print('Minimized pose written to : %s' % minimized_out)

    # Global docking
    v.dock(exhaustiveness=args.exhaustiveness, n_poses=args.n_poses)
    v.write_poses(args.out, n_poses=5, overwrite=True)
    print('Docked poses written to  : %s' % args.out)


if __name__ == '__main__':
    main()