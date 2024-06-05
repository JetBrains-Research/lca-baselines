 from pyscf import gto, scf, dmrgscf, mcscf, nevpt2, tools
mol = gto.Mole()
mol.build(
unit = 'angstrom',
atom = '''
Fe 0.000000 0.000000 0.000000
N 1.959000 0.000000 0.000000
N -1.959000 0.000000 0.000000
N 0.000000 1.959000 0.000000
N 0.000000 -1.959000 0.000000
''',
basis = 'def2-tzvp',
spin = 5,
charge = 0,
)

macro_sym_generator = tools.pyscf_to_molpro_symmeter(mol)
macro_sym_generator.irrep_name = {'A1': 'a', 'A2': 'b', 'E': 'e'}
macro_sym_generator.build_symmeter()
mol = macro_sym_generator.mol

m = scf.RHF(mol)
m.kernel()

dm = m.make_rdm1()
fe_dm = dm[:28, :28]
n_dm = dm[28:, 28:]

fe_mo = mol.unpack_lr_mo(m.mo_coeff, mol.irrep_id, range(28))
n_mo = mol.unpack_lr_mo(m.mo_coeff, mol.irrep_id, range(28, 38))

fe_act_space = fe_mo[:, :10]
n_act_space = n_mo[:, :2]

fe_act_space_sym = symmetrize_space_pyscf(fe_act_space, mol)
n_act_space_sym = symmetrize_space_pyscf(n_act_space, mol)

fe_act_space_sym_irrep = fe_act_space_sym.irrep_id
n_act_space_sym_irrep = n_act_space_sym.irrep_id

fe_act_space_sym_symb = macro_sym_generator.irrep_name[fe_act_space_sym_irrep[0]]
n_act_space_sym_symb = macro_sym_generator.irrep_name[n_act_space_sym_irrep[0]]

fe_act_space_sym_symb_and_num = fe_act_space_sym_symb + str(fe_act_space_sym.shape[1])
n_act_space_sym_symb_and_num = n_act_space_sym_symb + str(n_act_space_sym.shape[1])

fe_act_space_sym_symb_and_num_list = list(fe_act_space_sym_symb_and_num)
fe_act_space_sym_symb_and_num_list.append('1')
fe_act_space_sym_symb_and_num_list = tuple(fe_act_space_sym_symb_and_num_list)

n_act_space_sym_symb_and_num_list = list(n_act_space_sym_symb_and_num)
n_act_space_sym_symb_and_num_list.append('1')
n_act_space_sym_symb_and_num_list = tuple(n_act_space_sym_symb_and_num_list)

fe_act_space_sym_irrep_list = fe_act_space_sym_irrep.tolist()
n_act_space_sym_irrep_list = n_act_space_sym_irrep.tolist()

fe_act_space_sym_irrep_list_and_num = fe_act_space_sym_irrep_list + [10]
n_act_space_sym_irrep_list_and_num = n_act_space_sym_irrep_list + [2]

fe_act_space_sym_irrep_list_and_num_tuple = tuple(fe_act_space_sym_irrep_list_and_num)
n_act_space_sym_irrep_list_and_num_tuple = tuple(n_act_space_sym_irrep_list_and_num)

fe_act_space_sym_irrep_symb_list = [macro_sym_generator.irrep_name[i] for i in fe_act_space_sym_irrep_list]
n_act_space_sym_irrep_symb_list = [macro_sym_generator.irrep_name[i] for i in n_act_space_sym_irrep_list]

fe_act_space_sym_irrep_symb_list_and_num = fe_act_space_sym_irrep_symb_list + ['1']
n_act_space_sym_irrep_symb_list_and_num = n_act_space_sym_irrep_symb_list + ['1']

fe_act_space_sym_irrep_symb_list_and_num_tuple = tuple(fe_act_space_sym_irrep_symb_list_and_num)
n_act_space_sym_irrep_symb_list_and_num_tuple = tuple(n_act_space_sym_irrep_symb_list_and_num)

fe_act_space_sym_irrep_symb_list_str = ', '.join(fe_act_space_sym_irrep_symb_list)
n_act_space_sym_irrep_symb_list_str = ', '.join(n_act_space_sym_irrep_symb_list)

fe_act_space_sym_irrep_symb_list_str_and_num = fe_act_space_sym_irrep_symb_list_str + ', 1'
n_act_space_sym_irrep_symb_list_str_and_num = n_act_space_sym_irrep_symb_list_str + ', 1'

fe_act_space_sym_irrep_symb_list_str_and_num_tuple = tuple(fe_act_space_sym_irrep_symb_list_str_and_num.split(','))
n_act_space_sym_irrep_symb_list_str_and_num_tuple = tuple(n_act_space_sym_irrep_symb_list_str_and_num.split(','))

fe_act_space_sym_irrep_symb_list_str_and_num_tuple_str = ', '.join(fe_act_space_sym_irrep_symb_list_str_and_num_tuple)
n_act_space_sym_irrep_symb_list_str_and_num_tuple_str = ', '.join(n_act_space_sym_irrep_symb_list_str_and_num_tuple)

fe_act_space_sym_irrep_symb_list_str_and_num_tuple_str_list = list(fe_act_space_sym_irrep_symb_list_str_and_num_tuple_str.split(','))
fe_act_space_sym_irrep_symb_list_str_and_num_tuple_str_list[-1] = fe_act_space_sym_irrep_symb_list_str_and_num_tuple_str_list[-1][:-1]
fe_act_space_sym_irrep_symb_list_str_and_num_tuple_str_list = tuple(fe_act_space_sym_irrep_symb_list_str_and_num_tuple_str_list)

n_act_space_sym_irrep_symb_list_str_and_num_tuple_str_list = list(n_act_space_sym_irrep_symb_list_str_and_num_tuple_str.split(','))
n_act_space_