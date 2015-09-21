__author__ = 'bjimenez@bsc.es'

from prody import parsePDB, ANM, extendModel, saveAtoms, saveModel, sampleModes, writePDB, moveAtoms, \
    confProDy, Transformation, applyTransformation
import argparse
import os
import numpy as np
from conf.parameters import APP_NAME, MOVE_TO_CENTER, DECOYS_OUTPUT_FOLDER, \
    NUM_DECOYS, NUM_NORMAL_MODES, SAVE_MODELS, DEFAULT_RMSD, ROTATE, SCWRL_BIN
import conf.version as version
from util.mmath import get_random_rotation_matrix, get_affine
import multiprocessing as mp


def get_arguments():
    """Parses the command line arguments"""
    parser = argparse.ArgumentParser(prog=APP_NAME)
    parser.add_argument("pdb_structure", help="Original PDB structure")
    parser.add_argument("structure_name", help="Structure name")
    parser.add_argument("-d", "--num_decoys", help="Number of decoys to be generated",
                        dest="num_decoys", type=int, default=NUM_DECOYS)
    parser.add_argument("-r", "--rmsd", help="Ca RMSD between decoys",
                        dest="rmsd", type=float, default=DEFAULT_RMSD)
    parser.add_argument("-nm", "--normal_modes", help="Number of normal modes to consider",
                        dest="normal_modes", type=int, default=NUM_NORMAL_MODES)
    parser.add_argument("-tc", "--to_center", help="Move the structure to center of coordinates. By default, True",
                        dest="to_center", action='store_true', default=MOVE_TO_CENTER)
    parser.add_argument("-rr", "--random_rotation", help="Rotate the structure randomly. By default, False",
                        dest="random_rotation", action='store_true', default=ROTATE)
    parser.add_argument("-op", "--output_path", help="Output path folder",
                        dest="output_folder_path", default=DECOYS_OUTPUT_FOLDER)
    parser.add_argument("-sm", "--save_models", help="Save ProDy calculated models",
                        dest="save_models", type=bool, default=SAVE_MODELS)
    return parser.parse_args()


def get_logger():
    import logging
    logger = logging.getLogger(APP_NAME)
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('[%(name)s:%(levelname)s] %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger


def run_scwrl(decoy_file, output):
    fields = decoy_file.split('_')
    output_file_name = "%s_%s" % (fields[0], fields[2])
    cmd = '%s -i %s -o %s > /dev/null 2>&1' % (SCWRL_BIN, decoy_file, output_file_name)
    os.system(cmd)
    output.put(output_file_name)


if __name__ == "__main__":

    try:
        # Configure logging
        logger = get_logger()
        logger.info('Version %s' % version.number)
        confProDy(verbosity='info')

        args = get_arguments()

        protein = parsePDB(args.pdb_structure)
        logger.info('%s loaded' % args.pdb_structure)

        if args.to_center:
            logger.info('Moving original structure to the center')
            moveAtoms(protein, to=np.zeros(3), ag=True)

        if args.random_rotation:
            logger.info('Rotating the structure randomly')
            random_rotation_matrix = get_affine(get_random_rotation_matrix())
            random_rotation = Transformation(random_rotation_matrix)
            applyTransformation(random_rotation, protein)

        ca_atoms = protein.ca
        protein_anm = ANM('%s ca' % args.structure_name)
        protein_anm.buildHessian(ca_atoms)
        protein_anm.calcModes(n_modes=args.normal_modes)

        protein_anm_ext, protein_all = extendModel(protein_anm, ca_atoms, protein, norm=True)

        if args.save_models:
            saveAtoms(protein, args.structure_name)
            saveModel(protein_anm, args.structure_name)
            saveModel(protein_anm_ext, args.structure_name)

        ens = sampleModes(protein_anm_ext, atoms=protein.protein, n_confs=args.num_decoys, rmsd=args.rmsd)

        protein.addCoordset(ens.getCoordsets())
        protein.all.setBetas(0)
        protein.ca.setBetas(1)

        if not os.path.exists(args.output_folder_path):
            os.mkdir(args.output_folder_path)
        else:
            logger.warning('Folder %s already exists. Decoys may be overwritten.' % args.output_folder_path)

        decoys_generated = []
        for i in range(1, protein.numCoordsets()):
            file_name = os.path.join(args.output_folder_path,
                                     '%s_anm_%d.pdb' % (args.structure_name, i))
            writePDB(file_name, protein, csets=i)
            decoys_generated.append(file_name)
            logger.info('Decoy %s written' % file_name)

        logger.info('Minimizing side chains...')

        output = mp.Queue()
        processes = [mp.Process(target=run_scwrl, args=(decoy, output)) for decoy in decoys_generated]

        for p in processes:
            p.start()

        for p in processes:
            p.join()

        results = [output.get() for p in processes]

        logger.info('%d decoys generated.' % len(results))


    except KeyboardInterrupt:
        logger.error('Keyboard interrupt, bye.')
    except Exception, e:
        logger.error('Critical error. Reason: %s' % str(e))