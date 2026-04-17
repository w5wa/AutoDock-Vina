#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Vina CLI
#
# Usage examples:
#   Single ligand:
#     vina --receptor receptor.pdbqt --ligand ligand.pdbqt \
#          --center_x 15.0 --center_y 53.9 --center_z 16.9 \
#          --size_x 20 --size_y 20 --size_z 20
#
#   Batch from a folder (--input_dir / --dir):
#     vina --receptor receptor.pdbqt \
#          --center_x 15.0 --center_y 53.9 --center_z 16.9 \
#          --size_x 20 --size_y 20 --size_z 20 \
#          --input_dir ./ligands/ --dir ./results/
#
#   Split a multi-model PDBQT before docking:
#     vina --split multimodel.pdbqt --ligand_prefix lig_ --flex_prefix flex_
#

import argparse
import os
import sys

import numpy as np

from .vina import Vina
from . import utils


def cmd_lineparser():
    parser = argparse.ArgumentParser(
        description=(
            'AutoDock-Vina 1.2.0 (Python CLI)\n\n'
            'Molecular docking with the Vina, Vinardo, or AutoDock4 scoring function.\n'
            'Provide a receptor and one or more ligands, define the search box, and run.'
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            'Examples:\n'
            '  Single ligand docking:\n'
            '    vina -r receptor.pdbqt -l ligand.pdbqt \\\n'
            '         --center_x 15.0 --center_y 53.9 --center_z 16.9 \\\n'
            '         --size_x 20 --size_y 20 --size_z 20\n\n'
            '  Batch docking from input folder:\n'
            '    vina -r receptor.pdbqt \\\n'
            '         --center_x 15.0 --center_y 53.9 --center_z 16.9 \\\n'
            '         --size_x 20 --size_y 20 --size_z 20 \\\n'
            '         --input_dir ./ligands/ --dir ./results/\n\n'
            '  Split a multi-model PDBQT:\n'
            '    vina --split multimodel.pdbqt --ligand_prefix lig_ --flex_prefix flex_\n'
        )
    )

    # Split mode (standalone, mutually exclusive with docking)
    group_split = parser.add_argument_group(
        'Split options',
        'Split a multi-model PDBQT into individual ligand/flex files.'
    )
    group_split.add_argument('--split', dest='split', default=None,
                             type=str, action='store',
                             help='input multi-model PDBQT file to split')
    group_split.add_argument('--ligand_prefix', dest='ligand_prefix', default=None,
                             type=str, action='store',
                             help='output prefix for ligand files (default: <input>_ligand_)')
    group_split.add_argument('--flex_prefix', dest='flex_prefix', default=None,
                             type=str, action='store',
                             help='output prefix for flexible side-chain files (default: <input>_flex_)')

    # Input receptor, flex and ligand
    group_in = parser.add_argument_group('Input options')
    group_in.add_argument('-r', '--receptor', dest='receptor', default=None,
                          type=str, action='store',
                          help='rigid part of the receptor (PDBQT)')
    group_in.add_argument('-f', '--flex', dest='flex', default=None,
                          type=str, action='store',
                          help='flexible side chains, if any (PDBQT)')
    group_mut = group_in.add_mutually_exclusive_group()
    group_mut.add_argument('-l', '--ligand', dest='ligands', default=None,
                           nargs='+', action='store',
                           help='one or more ligand files (PDBQT)')
    group_mut.add_argument('-b', '--batch', dest='batch', default=None,
                           nargs='+', action='store',
                           help='list of ligand files for batch docking (PDBQT)')
    group_in.add_argument('--input_dir', dest='input_dir', default=None,
                          type=str, action='store',
                          help='directory containing ligand PDBQT files for batch docking '
                               '(alternative to --batch; all *.pdbqt files in the folder are used)')

    # Scoring function
    parser.add_argument('-s', '--scoring', dest='sf_name', default='vina',
                        type=str, choices=['ad4', 'vina', 'vinardo'], action='store',
                        help='scoring function: vina (default), vinardo, or ad4')

    # Search space / maps
    group_dim = parser.add_argument_group('Search space options')
    group_dim.add_argument('-m', '--maps', dest='maps', default=None,
                           action='store',
                           help='affinity maps prefix for ad4, vina, or vinardo scoring function')
    group_dim.add_argument('--center_x', dest='center_x', default=None,
                           type=float, action='store',
                           help='X coordinate of the box center (Angstrom)')
    group_dim.add_argument('--center_y', dest='center_y', default=None,
                           type=float, action='store',
                           help='Y coordinate of the box center (Angstrom)')
    group_dim.add_argument('--center_z', dest='center_z', default=None,
                           type=float, action='store',
                           help='Z coordinate of the box center (Angstrom)')
    group_dim.add_argument('--size_x', dest='size_x', default=None,
                           type=float, action='store',
                           help='box size along X (Angstrom)')
    group_dim.add_argument('--size_y', dest='size_y', default=None,
                           type=float, action='store',
                           help='box size along Y (Angstrom)')
    group_dim.add_argument('--size_z', dest='size_z', default=None,
                           type=float, action='store',
                           help='box size along Z (Angstrom)')

    # Actions
    group_actions = parser.add_argument_group('Action options')
    group_actions.add_argument('--randomize_only', dest='randomize_only', default=False,
                               action='store_true',
                               help='randomize input pose, attempting to avoid clashes')
    group_actions.add_argument('--score_only', dest='score_only', default=False,
                               action='store_true',
                               help='score the current pose without docking (requires maps)')
    group_actions.add_argument('--local_only', dest='local_only', default=False,
                               action='store_true',
                               help='perform local search only (no global docking)')

    # Output
    group_out = parser.add_argument_group('Output options')
    group_out.add_argument('-o', '--out', dest='out', default=None,
                           type=str, action='store',
                           help='output PDBQT file (default: derived from ligand filename)')
    group_out.add_argument('-d', '--dir', dest='dir', default=None,
                           type=str, action='store',
                           help='output directory for batch/input_dir mode')
    group_out.add_argument('--write_maps', dest='write_maps', default=None,
                           type=str, action='store',
                           help='write affinity maps to this prefix (directory + prefix name)')

    # Extra
    group_extra = parser.add_argument_group('Extra options')
    group_extra.add_argument('--cpu', dest='cpu', default=0,
                             type=int, action='store',
                             help='number of CPUs to use (0 = auto-detect)')
    group_extra.add_argument('--seed', dest='seed', default=0,
                             type=int, action='store',
                             help='explicit random seed (0 = random)')
    group_extra.add_argument('--exhaustiveness', dest='exhaustiveness', default=8,
                             type=int, action='store',
                             help='exhaustiveness of global search (higher = slower but more thorough): 1+')
    group_extra.add_argument('--max_evals', dest='max_evals', default=0,
                             type=int, action='store',
                             help='number of evaluations per MC run (0 = heuristic)')
    group_extra.add_argument('--num_modes', dest='num_modes', default=9,
                             type=int, action='store',
                             help='maximum number of binding modes to generate')
    group_extra.add_argument('--min_rmsd', dest='min_rmsd', default=1.0,
                             type=float, action='store',
                             help='minimum RMSD between output poses (Angstrom)')
    group_extra.add_argument('--energy_range', dest='energy_range', default=3.0,
                             type=float, action='store',
                             help='maximum energy difference between best and worst displayed pose (kcal/mol)')
    group_extra.add_argument('--spacing', dest='grid_spacing', default=0.375,
                             type=float, action='store',
                             help='grid spacing (Angstrom)')
    group_extra.add_argument('--verbosity', dest='verbosity', default=1,
                             type=int, action='store',
                             help='verbosity: 0=silent, 1=normal, 2=verbose')

    # Scoring function weights
    group_weight = parser.add_argument_group('Scoring function weight options')
    group_weight.add_argument('--weight_gauss1', dest='weight_gauss1', default=-0.035579,
                              type=float, action='store', help='Vina gauss_1 weight')
    group_weight.add_argument('--weight_gauss2', dest='weight_gauss2', default=-0.005156,
                              type=float, action='store', help='Vina gauss_2 weight')
    group_weight.add_argument('--weight_repulsion', dest='weight_repulsion', default=0.840245,
                              type=float, action='store', help='Vina repulsion weight')
    group_weight.add_argument('--weight_hydrophobic', dest='weight_hydrophobic', default=-0.035069,
                              type=float, action='store', help='Vina hydrophobic weight')
    group_weight.add_argument('--weight_hydrogen', dest='weight_hydrogen', default=-0.587439,
                              type=float, action='store', help='Vina hydrogen bond weight')
    group_weight.add_argument('--weight_rot', dest='weight_rot', default=0.05846,
                              type=float, action='store', help='Vina N_rot weight')
    group_weight.add_argument('--weight_vinardo_gauss1', dest='weight_vinardo_gauss1', default=-0.045,
                              type=float, action='store', help='Vinardo gauss_1 weight')
    group_weight.add_argument('--weight_vinardo_repulsion', dest='weight_vinardo_repulsion', default=0.8,
                              type=float, action='store', help='Vinardo repulsion weight')
    group_weight.add_argument('--weight_vinardo_hydrophobic', dest='weight_vinardo_hydrophobic', default=-0.035,
                              type=float, action='store', help='Vinardo hydrophobic weight')
    group_weight.add_argument('--weight_vinardo_hydrogen', dest='weight_vinardo_hydrogen', default=-0.600,
                              type=float, action='store', help='Vinardo hydrogen bond weight')
    group_weight.add_argument('--weight_vinardo_rot', dest='weight_vinardo_rot', default=0.05846,
                              type=float, action='store', help='Vinardo N_rot weight')
    group_weight.add_argument('--weight_ad4_vdw', dest='weight_ad4_vdw', default=0.1662,
                              type=float, action='store', help='AD4 vdW weight')
    group_weight.add_argument('--weight_ad4_hb', dest='weight_ad4_hb', default=0.1209,
                              type=float, action='store', help='AD4 H-bond weight')
    group_weight.add_argument('--weight_ad4_elec', dest='weight_ad4_elec', default=0.1406,
                              type=float, action='store', help='AD4 electrostatic weight')
    group_weight.add_argument('--weight_ad4_desolv', dest='weight_ad4_desolv', default=0.1322,
                              type=float, action='store', help='AD4 desolvation weight')
    group_weight.add_argument('--weight_ad4_rot', dest='weight_ad4_rot', default=0.2983,
                              type=float, action='store', help='AD4 torsional weight')
    group_weight.add_argument('--weight_glue', dest='weight_glue', default=50.0,
                              type=float, action='store', help='macrocycle glue weight')

    args = parser.parse_args()

    # --- Split mode: if --split is given, run only the splitter and exit early ---
    if args.split is not None:
        return args

    # --- Expand --input_dir into --batch ---
    if args.input_dir is not None:
        if not os.path.isdir(args.input_dir):
            parser.error('ERROR: --input_dir %s does not exist or is not a directory.' % args.input_dir)
        pdbqt_files = sorted(
            f for f in os.listdir(args.input_dir) if f.lower().endswith('.pdbqt')
        )
        if not pdbqt_files:
            parser.error('ERROR: No PDBQT files found in --input_dir %s.' % args.input_dir)
        args.batch = [os.path.join(args.input_dir, f) for f in pdbqt_files]
        args.ligands = None

    # We need the rigid receptor or the maps, not both
    if args.receptor is not None and args.maps is not None:
        parser.error('ERROR: Cannot specify both --receptor and --maps at the same time.')

    if args.maps is None:
        if not all(v is not None for v in [args.center_x, args.center_y, args.center_z]):
            parser.error(
                'ERROR: The center of the grid box is not fully defined '
                '(X=%s Y=%s Z=%s). Provide --center_x, --center_y, and --center_z.'
                % (args.center_x, args.center_y, args.center_z)
            )
        if not all(v is not None for v in [args.size_x, args.size_y, args.size_z]):
            parser.error(
                'ERROR: The size of the grid box is not fully defined '
                '(X=%s Y=%s Z=%s). Provide --size_x, --size_y, and --size_z.'
                % (args.size_x, args.size_y, args.size_z)
            )

    # Scoring-function/receptor compatibility
    if args.sf_name in ('vina', 'vinardo'):
        if args.receptor is None and args.maps is None:
            parser.error('ERROR: Either --receptor or --maps must be specified for %s scoring.' % args.sf_name)
    elif args.sf_name == 'ad4':
        if args.receptor is not None:
            parser.error('ERROR: --receptor is not allowed with the AD4 scoring function (use --flex).')
        if args.maps is None:
            parser.error('ERROR: --maps must be specified for the AD4 scoring function.')

    # Ligand / batch requirements
    if args.batch is None and args.ligands is None:
        parser.error('ERROR: No ligand specified. Use --ligand, --batch, or --input_dir.')

    if args.ligands is not None:
        if args.dir is not None:
            parser.error('ERROR: --dir cannot be used in --ligand mode; use --out instead.')
        if args.out is None and not args.score_only:
            if len(args.ligands) == 1:
                args.out = '%s_out.pdbqt' % os.path.splitext(args.ligands[0])[0]
            else:
                parser.error(
                    'ERROR: --out must be specified when docking multiple ligands simultaneously.'
                )
    elif args.batch is not None:
        if args.dir is None:
            parser.error('ERROR: --dir must be specified for batch/input_dir mode.')
        if not os.path.isdir(args.dir):
            parser.error('ERROR: Output directory %s does not exist.' % args.dir)

    return args


def main():
    args = cmd_lineparser()

    # --- Split mode ---
    if args.split is not None:
        ligand_prefix = args.ligand_prefix
        flex_prefix = args.flex_prefix
        n_written = utils.split_pdbqt(args.split, ligand_prefix, flex_prefix)
        if args.verbosity > 0:
            print('Split %d model(s) from %s.' % (n_written, args.split))
        return

    # --- Verbosity header ---
    if args.verbosity > 0:
        print('Scoring function : %s' % args.sf_name)
        if args.receptor is not None:
            print('Rigid receptor   : %s' % args.receptor)
        if args.flex is not None:
            print('Flex receptor    : %s' % args.flex)

        if args.ligands is not None:
            if len(args.ligands) == 1:
                print('Ligand           : %s' % args.ligands[0])
            else:
                print('Ligands          :')
                for ligand in args.ligands:
                    print('  - %s' % ligand)
        elif args.batch is not None:
            print('Ligands (batch)  : %d molecules' % len(args.batch))

        if args.maps is None:
            print('Center           : X %.3f  Y %.3f  Z %.3f' % (args.center_x, args.center_y, args.center_z))
            print('Size             : X %.2f  Y %.2f  Z %.2f' % (args.size_x, args.size_y, args.size_z))
            print('Grid spacing     : %.3f Angstrom' % args.grid_spacing)

        print('Exhaustiveness   : %d' % args.exhaustiveness)
        print('CPU              : %d' % args.cpu)
        if args.seed != 0:
            print('Seed             : %d' % args.seed)
        print('Verbosity        : %d' % args.verbosity)

    v = Vina(args.sf_name, args.cpu, args.seed, verbosity=args.verbosity)

    # Set receptor (rigid_name can be ignored for AD4)
    if args.receptor is not None or args.flex is not None:
        v.set_receptor(args.receptor, args.flex)

    # Set scoring function weights (always explicit so custom weights are honoured)
    if args.sf_name == 'vina':
        v.set_weights([args.weight_gauss1, args.weight_gauss2, args.weight_repulsion,
                       args.weight_hydrophobic, args.weight_hydrogen, args.weight_glue,
                       args.weight_rot])
    elif args.sf_name == 'vinardo':
        v.set_weights([args.weight_vinardo_gauss1, args.weight_vinardo_repulsion,
                       args.weight_vinardo_hydrophobic, args.weight_vinardo_hydrogen,
                       args.weight_glue, args.weight_vinardo_rot])
    else:  # ad4
        v.set_weights([args.weight_ad4_vdw, args.weight_ad4_hb, args.weight_ad4_elec,
                       args.weight_ad4_desolv, args.weight_glue, args.weight_ad4_rot])
        v.load_maps(args.maps)
        if args.write_maps is not None:
            v.write_maps(args.write_maps)

    # --- Helper: set up maps for vina/vinardo ---
    def _setup_vina_maps():
        if args.sf_name in ('vina', 'vinardo'):
            if args.maps is not None:
                v.load_maps(args.maps)
            else:
                box_center = (args.center_x, args.center_y, args.center_z)
                box_size = (args.size_x, args.size_y, args.size_z)
                v.compute_vina_maps(box_center, box_size, args.grid_spacing)
                if args.write_maps is not None:
                    v.write_maps(args.write_maps)

    # --- Helper: run docking / scoring action ---
    def _run_action(out_name):
        if args.randomize_only:
            v.randomize()
            v.write_pose(out_name, overwrite=True)
        elif args.score_only:
            energies = v.score()
            if args.verbosity > 0:
                print('Score: %.3f kcal/mol' % energies[0])
        elif args.local_only:
            v.optimize()
            v.write_pose(out_name, overwrite=True)
        else:
            v.dock(args.exhaustiveness, args.num_modes, args.min_rmsd, args.max_evals)
            v.write_poses(out_name, args.num_modes, args.energy_range, overwrite=True)

    if args.ligands is not None:
        v.set_ligand_from_file(args.ligands)
        _setup_vina_maps()
        _run_action(args.out)
    else:
        _setup_vina_maps()
        for ligand in args.batch:
            v.set_ligand_from_file(ligand)
            molecule_name = os.path.splitext(os.path.basename(ligand))[0]
            out_name = os.path.join(args.dir, '%s_out.pdbqt' % molecule_name)
            _run_action(out_name)


if __name__ == '__main__':
    main()
