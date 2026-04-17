#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Vina - utils
#

import os


def check_file_writable(fnm):
    """Source: https://www.novixys.com/blog/python-check-file-can-read-write/"""
    if os.path.exists(fnm):
        # path exists
        if os.path.isfile(fnm): # is it a file or a dir?
            # also works when file is a link and the target is writable
            return os.access(fnm, os.W_OK)
        else:
            return False # path is a dir, so cannot write as a file
    # target does not exist, check perms on parent dir
    pdir = os.path.dirname(fnm)
    if not pdir: pdir = '.'
    # target is creatable if parent dir is writable
    return os.access(pdir, os.W_OK)


def _default_prefix(input_name, suffix):
    """Return <input_name_without_.pdbqt><suffix>."""
    base = input_name
    if base.lower().endswith('.pdbqt'):
        base = base[:-6]
    return base + suffix


def split_pdbqt(input_file, ligand_prefix=None, flex_prefix=None):
    """Split a multi-model PDBQT file produced by AutoDock Vina into individual files.

    Each MODEL/ENDMDL block is written as a separate numbered file.
    Lines between BEGIN_RES / END_RES are treated as flexible residue (flex) data;
    all other lines inside a MODEL block are treated as ligand data.

    Args:
        input_file (str): path to the multi-model PDBQT file.
        ligand_prefix (str or None): filename prefix for ligand output files.
            Defaults to ``<input_file_without_extension>_ligand_``.
        flex_prefix (str or None): filename prefix for flex output files.
            Defaults to ``<input_file_without_extension>_flex_``.

    Returns:
        int: number of models written.

    Raises:
        FileNotFoundError: if *input_file* does not exist.
        ValueError: if the PDBQT file is malformed (misplaced MODEL/ENDMDL/BEGIN_RES/END_RES tags).
    """
    if not os.path.exists(input_file):
        raise FileNotFoundError('Error: input file %s does not exist.' % input_file)

    if ligand_prefix is None:
        ligand_prefix = _default_prefix(input_file, '_ligand_')
    if flex_prefix is None:
        flex_prefix = _default_prefix(input_file, '_flex_')

    # Parse ---------------------------------------------------------------
    models = []        # list of {'ligand': [lines], 'flex': [lines]}
    current = None
    in_flex = False

    with open(input_file, 'r') as fh:
        for lineno, line in enumerate(fh, start=1):
            line = line.rstrip('\n')
            stripped = line.strip()

            if stripped.startswith('MODEL'):
                if current is not None:
                    raise ValueError(
                        'Misplaced MODEL tag at line %d (previous MODEL not closed with ENDMDL).' % lineno
                    )
                current = {'ligand': [], 'flex': []}
                in_flex = False

            elif stripped.startswith('ENDMDL'):
                if current is None:
                    raise ValueError('Misplaced ENDMDL tag at line %d (no open MODEL).' % lineno)
                if in_flex:
                    raise ValueError(
                        'ENDMDL at line %d inside a BEGIN_RES/END_RES block.' % lineno
                    )
                models.append(current)
                current = None

            elif stripped.startswith('BEGIN_RES'):
                if current is None:
                    raise ValueError('BEGIN_RES outside MODEL at line %d.' % lineno)
                if in_flex:
                    raise ValueError('Nested BEGIN_RES at line %d.' % lineno)
                in_flex = True
                current['flex'].append(line)

            elif stripped.startswith('END_RES'):
                if current is None or not in_flex:
                    raise ValueError('Misplaced END_RES at line %d.' % lineno)
                in_flex = False
                current['flex'].append(line)

            else:
                if current is None:
                    # Skip blank lines / comments outside MODEL blocks
                    if stripped and not stripped.startswith('REMARK'):
                        raise ValueError(
                            'Non-empty line outside MODEL block at line %d: %s' % (lineno, line)
                        )
                    continue
                if in_flex:
                    current['flex'].append(line)
                else:
                    current['ligand'].append(line)

    if current is not None:
        raise ValueError('Input file ended without a closing ENDMDL tag.')

    # Write ---------------------------------------------------------------
    n_models = len(models)
    width = len(str(n_models))
    fmt = '%s%0*d.pdbqt'

    for idx, m in enumerate(models, start=1):
        suffix_num = idx
        if m['ligand']:
            out_path = fmt % (ligand_prefix, width, suffix_num)
            with open(out_path, 'w') as fh:
                fh.write('\n'.join(m['ligand']) + '\n')
        if m['flex']:
            out_path = fmt % (flex_prefix, width, suffix_num)
            with open(out_path, 'w') as fh:
                fh.write('\n'.join(m['flex']) + '\n')

    return n_models
