#-------------------------------------------------------------------------
#
# Copyright (c) 2009, IMB, RWTH Aachen.
# All rights reserved.
#
# This software is provided without warranty under the terms of the BSD
# license included in simvisage/LICENSE.txt and may be redistributed only
# under the conditions described in the aforementioned license.  The license
# is also available online at http://www.simvisage.com/licenses/BSD.txt
#
# Thanks for using Simvisage open source!
#
# Created on Nov 8, 2011 by: rch

import imp
import operator
import os
import platform
import time

from traits.api import HasStrictTraits, Property, cached_property, \
    List, Str, Int, Trait, Bool, Interface, provides

import numpy as np  # import numpy package
import scipy.stats.distributions as distr  # import distributions
import util.weave as weave

from stats.spirrid_bak import CodeGen
from stats.spirrid_bak import RV


if platform.system() == 'Linux':
    sysclock = time.time
elif platform.system() == 'Windows':
    sysclock = time.clock


class ICodeGenLangDict(Interface):
    pass


@provides(ICodeGenLangDict)
class CodeGenLangDictC(HasStrictTraits):

    LD_BEGIN_EPS_LOOP_ACTIVE = 'for( int i_eps = 0; i_eps < %(i)i; i_eps++){\n'
    LD_END_EPS_LOOP_ACTIVE = '};\n'
    LD_ACCESS_EPS_IDX = '\tdouble eps = e_arr( i_eps );\n'
    LD_ACCESS_EPS_PTR = '\tdouble eps = *( e_arr + i_eps );\n'
    LD_ASSIGN_EPS = 'double eps = e;\n'

    LD_LINE_MACRO = '#line 100\n'
    LD_INIT_MU_Q = 'double mu_q=0;\n'
    LD_INIT_Q = 'double q=0;\n'
    LD_EVAL_MU_Q = '\tmu_q +=  q * dG;\n'
    LD_ADD_MU_Q = '\tmu_q += q;\n'
    LD_ASSIGN_MU_Q_IDX = 'mu_q_arr(i_eps) = mu_q;\n'
    LD_ASSIGN_MU_Q_PTR = '*(mu_q_arr + i_eps) = mu_q;\n'
    LD_RETURN_MU_Q = 'return_val = mu_q;'

    LD_INIT_THETA = 'double %s = %g;\n'
    LD_BEGIN_THETA_DEEP_LOOP = '%(t)sfor( int i_%(s)s = 0; i_%(s)s < %(i)i; i_%(s)s++){\n'
    LD_END_THETA_DEEP_LOOP = '};\n'
    LD_ACCESS_THETA_IDX = '\tdouble %s = %s_flat( i_%s );\n'
    LD_ACCESS_THETA_PTR = '\tdouble %s = *( %s_flat + i_%s );\n'

    LD_BEGIN_THETA_FLAT_LOOP = 'for( int i = 0; i < %i; i++){\n'
    LD_ACCESS_THETA_FLAT_IDX = '\tdouble %s = %s_flat( i);\n'
    LD_ACCESS_THETA_FLAT_PTR = '\tdouble %s = *( %s_flat + i );\n'
    LD_END_THETA_FLAT_LOOP = '};\n'

    LD_DECLARE_DG_IDX = '\tdouble dG = dG_grid('
    LD_DECLARE_DG_PTR = '\tdouble dG = '
    LD_ACCESS_DG_IDX = 'i_%s'
    LD_ACCESS_DG_PTR = '*( %s_dG + i_%s)'
    LD_ASSIGN_DG = 'double dG = %g;\n'
    LD_END_BRACE = ');\n'


@provides(ICodeGenLangDict)
class CodeGenLangDictCython(HasStrictTraits):

    LD_BEGIN_EPS_LOOP_ACTIVE = '\tcdef np.ndarray mu_q_arr = np.zeros_like( e_arr )\n\tfor i_eps from 0 <= i_eps < %(i)i:\n\t\teps = e_arr[i_eps]\n'
    LD_END_EPS_LOOP_ACTIVE = '\n'
    LD_ACCESS_EPS_IDX = ''
    LD_ACCESS_EPS_PTR = ''
    LD_ASSIGN_EPS = ''

    LD_LINE_MACRO = ''
    LD_INIT_MU_Q = ''
    LD_INIT_Q = '\tmu_q = 0\n'
    LD_EVAL_MU_Q = 'mu_q += q * dG\n'
    LD_ADD_MU_Q = 'mu_q += q\n'
    LD_ASSIGN_MU_Q_IDX = '\t\tmu_q_arr[i_eps] = mu_q\n\treturn mu_q_arr\n'
    LD_ASSIGN_MU_Q_PTR = '\t\tmu_q_arr[i_eps] = mu_q\n\treturn mu_q_arr\n'
    LD_RETURN_MU_Q = '\treturn mu_q'

    LD_INIT_THETA = '\t\t%s = %g\n'
    LD_BEGIN_THETA_DEEP_LOOP = '%(t)s\tfor i_%(s)s from 0 <= i_%(s)s <%(i)i:\n'
    LD_END_THETA_DEEP_LOOP = '\n'
    LD_ACCESS_THETA_IDX = '\t\t%s = %s_flat[ i_%s ]\n'
    LD_ACCESS_THETA_PTR = '\t\t%s = %s_flat[ i_%s ]\n'

    LD_BEGIN_THETA_FLAT_LOOP = '\tfor i from 0 <= i < %i:\n'
    LD_ACCESS_THETA_FLAT_IDX = '\t\t%s = %s_flat[i]\n'
    LD_ACCESS_THETA_FLAT_PTR = '\t\t%s = %s_flat[i]\n'
    LD_END_THETA_FLAT_LOOP = '\n'

    LD_DECLARE_DG_IDX = '%(t)s\tdG = dG_grid['
    LD_DECLARE_DG_PTR = '%(t)s\tdG = '
    LD_ACCESS_DG_IDX = 'i_%s'
    LD_ACCESS_DG_PTR = '%s_dG[i_%s]'
    LD_ASSIGN_DG = '\tdG = %g\n'
    LD_END_BRACE = ']\n'

#=========================================================================
# Generator of the C-code
#=========================================================================


class CodeGenCompiled(CodeGen):
    '''
        C-code is generated using the inline feature of scipy.
    '''
    #=========================================================================
    # Inspection of the randomization - needed by CodeGenCompiled
    #=========================================================================
    evar_names = Property(depends_on='q, recalc')

    @cached_property
    def _get_evar_names(self):
        return self.spirrid.evar_names

    var_names = Property(depends_on='q, recalc')

    @cached_property
    def _get_var_names(self):
        return self.spirrid.tvar_names

    # count the random variables
    n_rand_vars = Property(depends_on='tvars, recalc')

    @cached_property
    def _get_n_rand_vars(self):
        dt = list(map(type, self.spirrid.tvar_lst))
        return dt.count(RV)

    # get the indexes of the random variables within the parameter list
    rand_var_idx_list = Property(depends_on='tvars, recalc')

    @cached_property
    def _get_rand_var_idx_list(self):
        dt = np.array(list(map(type, self.spirrid.tvar_lst)))
        return np.where(dt == RV)[0]

    # get the names of the random variables
    rand_var_names = Property(depends_on='tvars, recalc')

    @cached_property
    def _get_rand_var_names(self):
        return self.var_names[self.rand_var_idx_list]

    # get the randomization arrays
    theta_arrs = Property(List, depends_on='tvars, recalc')

    @cached_property
    def _get_theta_arrs(self):
        rv_getter = operator.itemgetter(*self.rand_var_idx_list)
        theta = self.spirrid.sampling.theta
        return [x.flatten() for x in rv_getter(theta)]

    # get the randomization arrays
    dG_arrs = Property(List, depends_on='tvars, recalc')

    @cached_property
    def _get_dG_arrs(self):
        dG_ogrid = self.spirrid.sampling.dG_ogrid
        rv_getter = operator.itemgetter(*self.rand_var_idx_list)
        return [x.flatten() for x in rv_getter(dG_ogrid)]

    arg_names = Property(
        depends_on='rf_change, rand_change, +codegen_option, recalc')

    @cached_property
    def _get_arg_names(self):

        arg_names = []
        # create argument string for inline function
        if self.compiled_eps_loop:
            # @todo: e_arr must be evar_names
            arg_names += ['mu_q_arr', 'e_arr']
        else:
            arg_names.append('e')

        arg_names += ['%s_flat' % name for name in self.rand_var_names]

        arg_names += self._get_arg_names_dG()

        return arg_names

    ld = Trait('c', dict(c=CodeGenLangDictC(),
                         cython=CodeGenLangDictCython()))

    #=========================================================================
    # Configuration of the code
    #=========================================================================
    #
    # compiled_eps_loop:
    # If set True, the loop over the control variable epsilon is compiled
    # otherwise, python loop is used.
    compiled_eps_loop = Bool(False, codegen_option=True)

    #=========================================================================
    # compiled_eps_loop - dependent code
    #=========================================================================

    compiled_eps_loop_feature = Property(
        depends_on='compiled_eps_loop, recalc')

    @cached_property
    def _get_compiled_eps_loop_feature(self):
        if self.compiled_eps_loop == True:
            return self.ld_.LD_BEGIN_EPS_LOOP_ACTIVE, self.ld_.LD_END_EPS_LOOP_ACTIVE
        else:
            return self.ld_.LD_ASSIGN_EPS, ''

    LD_BEGIN_EPS_LOOP = Property

    def _get_LD_BEGIN_EPS_LOOP(self):
        return self.compiled_eps_loop_feature[0]

    LD_END_EPS_LOOP = Property

    def _get_LD_END_EPS_LOOP(self):
        return self.compiled_eps_loop_feature[1]

    #
    # cached_dG:
    # If set to True, the cross product between the pdf values of all random variables
    # will be precalculated and stored in an n-dimensional grid
    # otherwise the product is performed for every epsilon in the inner loop anew
    #
    cached_dG = Bool(True, codegen_option=True)

    #=========================================================================
    # cached_dG - dependent code
    #=========================================================================
    cached_dG_feature = Property(depends_on='cached_dG, recalc')

    @cached_property
    def _get_cached_dG_feature(self):
        if self.compiled_eps_loop:
            if self.cached_dG == True:
                return self.ld_.LD_ACCESS_EPS_IDX, self.ld_.LD_ACCESS_THETA_IDX, self.ld_.LD_ASSIGN_MU_Q_IDX
            else:
                return self.ld_.LD_ACCESS_EPS_PTR, self.ld_.LD_ACCESS_THETA_PTR, self.ld_.LD_ASSIGN_MU_Q_PTR
        else:
            if self.cached_dG == True:
                return self.ld_.LD_ACCESS_EPS_IDX, self.ld_.LD_ACCESS_THETA_IDX, self.ld_.LD_ASSIGN_MU_Q_IDX
            else:
                return self.ld_.LD_ACCESS_EPS_PTR, self.ld_.LD_ACCESS_THETA_PTR, self.ld_.LD_ASSIGN_MU_Q_PTR

    LD_ACCESS_EPS = Property

    def _get_LD_ACCESS_EPS(self):
        return self.cached_dG_feature[0]

    LD_ACCESS_THETA = Property

    def _get_LD_ACCESS_THETA(self):
        return '%s' + self.cached_dG_feature[1]

    LD_ASSIGN_MU_Q = Property

    def _get_LD_ASSIGN_MU_Q(self):
        return self.cached_dG_feature[2]

    LD_N_TAB = Property

    def _get_LD_N_TAB(self):
        if self.spirrid.sampling_type == 'LHS' or self.spirrid.sampling_type == 'MCS':
            if self.compiled_eps_loop:
                return 3
            else:
                return 2
        else:
            if self.compiled_eps_loop:
                return self.n_rand_vars + 2
            else:
                return self.n_rand_vars + 1

    #-------------------------------------------------------------------------
    # Configurable generation of C-code for the mean curve evaluation
    #-------------------------------------------------------------------------
    code = Property(
        depends_on='rf_change, rand_change, +codegen_option, eps_change, recalc')

    @cached_property
    def _get_code(self):

        code_str = ''
        if self.compiled_eps_loop:

            # create code string for inline function
            #
            n_eps = len(self.spirrid.evar_lst[0])
            code_str += self.LD_BEGIN_EPS_LOOP % {'i': n_eps}
            code_str += self.LD_ACCESS_EPS

        else:

            # create code string for inline function
            #
            code_str += self.ld_.LD_ASSIGN_EPS

        code_str += self.ld_.LD_INIT_MU_Q
        if self.compiled_eps_loop:
            code_str += '\t' + self.ld_.LD_INIT_Q
        else:
            code_str += self.ld_.LD_INIT_Q
        code_str += self.ld_.LD_LINE_MACRO
        # create code for constant params
        for name, distr in zip(self.var_names, self.spirrid.tvar_lst):
            if type(distr) is float:
                code_str += self.ld_.LD_INIT_THETA % (name, distr)

        code_str += self._get_code_dG_declare()

        inner_code_str = ''
        lang = self.ld + '_code'
        q_code = getattr(self.spirrid.q, lang)
        import textwrap
        q_code = textwrap.dedent(q_code)
        q_code_split = q_code.split('\n')
        for i, s in enumerate(q_code_split):
            q_code_split[i] = self.LD_N_TAB * '\t' + s
        q_code = '\n'.join(q_code_split)

        if self.n_rand_vars > 0:
            inner_code_str += self._get_code_dG_access()
            inner_code_str += q_code + '\n' + \
                (self.LD_N_TAB + 1) * '\t' + self.ld_.LD_EVAL_MU_Q
        else:
            inner_code_str += q_code + \
                self.ld_.LD_ADD_MU_Q

        code_str += self._get_code_inner_loops(inner_code_str)

        if self.compiled_eps_loop:
            if self.cached_dG:  # blitz matrix
                code_str += self.ld_.LD_ASSIGN_MU_Q_IDX
            else:
                code_str += self.ld_.LD_ASSIGN_MU_Q_PTR
            code_str += self.LD_END_EPS_LOOP
        else:
            code_str += self.ld_.LD_RETURN_MU_Q
        return code_str

    compiler_verbose = Int(1)
    compiler = Str('gcc')

    def get_code(self):
        if self.ld == 'c':
            return self.get_c_code()
        elif self.ld == 'cython':
            return self.get_cython_code()

    def get_cython_code(self):

        cython_header = 'import numpy as np\ncimport numpy as np\nctypedef np.double_t DTYPE_t\ncimport cython\n\n@cython.boundscheck(False)\n@cython.wraparound(False)\n@cython.cdivision(True)\ndef mu_q(%s):\n\tcdef double mu_q\n'
        #@todo - for Cython cdef variables and generalize function def()
        arg_values = {}
        for name, theta_arr in zip(self.rand_var_names, self.theta_arrs):
            arg_values['%s_flat' % name] = theta_arr
        arg_values.update(self._get_arg_values_dG())

        DECLARE_ARRAY = 'np.ndarray[DTYPE_t, ndim=1] '
        def_dec = DECLARE_ARRAY + 'e_arr'
        def_dec += ',' + DECLARE_ARRAY
        def_dec += (',' + DECLARE_ARRAY).join(arg_values)

        cython_header = cython_header % def_dec
        cython_header += '    cdef double '
        cython_header += ', '.join(self.var_names) + ', eps, dG, q\n'
        cython_header += '    cdef int i_'
        cython_header += ', i_'.join(self.var_names) + '\n'
        if self.cached_dG:
            cython_header = cython_header.replace(
                r'1] dG_grid', r'%i] dG_grid' % self.n_rand_vars)
        if self.compiled_eps_loop == False:
            cython_header = cython_header.replace(
                r'np.ndarray[DTYPE_t, ndim=1] e_arr', r'double e_arr')
            cython_header = cython_header.replace(r'eps,', r'eps = e_arr,')
        cython_code = (cython_header + self.code).replace('\t', '    ')
        cython_file_name = 'spirrid_cython.pyx'

        print('checking for previous cython code')
        regenerate_code = True
        if os.path.exists(cython_file_name):
            f_in = open(cython_file_name, 'r').read()
            if f_in == cython_code:
                regenerate_code = False

        if regenerate_code:
            infile = open('spirrid_cython.pyx', 'w')
            infile.write(cython_code)
            infile.close()
            print('pyx file updated')

        import pyximport
        t = sysclock()
        pyximport.install(reload_support=True)
        import spirrid_cython
        if regenerate_code:
            imp.reload(spirrid_cython)
        print(('>>> pyximport', sysclock() - t))
        mu_q = spirrid_cython.mu_q

        def mu_q_method(eps):
            if self.compiled_eps_loop:
                args = {'e_arr': eps}
                args.update(arg_values)
                mu_q_arr = mu_q(**args)
            else:
                # Python loop over eps
                #
                mu_q_arr = np.zeros_like(eps)
                for idx, e in enumerate(eps):
                    # C loop over random dimensions
                    #
                    arg_values['e_arr'] = e  # prepare the parameter
                    mu_q_val = mu_q(**arg_values)
                    # add the value to the return array
                    mu_q_arr[idx] = mu_q_val
            return mu_q_arr, None
        return mu_q_method

    def get_c_code(self):
        '''
            Return the code for the given sampling of the rand domain.
        '''
        def mu_q_method(e):
            '''Template for the evaluation of the mean response.
            '''
            self._set_compiler()

            compiler_args, linker_args = self.extra_args
            print(compiler_args)

            # prepare the array of the control variable discretization
            #
            eps_arr = e
            mu_q_arr = np.zeros_like(eps_arr)

            # prepare the parameters for the compiled function in
            # a separate dictionary
            arg_values = {}

            if self.compiled_eps_loop:

                # for compiled eps_loop the whole input and output array must be passed to c
                #
                arg_values['e_arr'] = eps_arr
                arg_values['mu_q_arr'] = mu_q_arr

            # prepare the lengths of the arrays to set the iteration bounds
            #
            for name, theta_arr in zip(self.rand_var_names, self.theta_arrs):
                arg_values['%s_flat' % name] = theta_arr

            arg_values.update(self._get_arg_values_dG())

            if self.cached_dG:
                conv = weave.converters.blitz
            else:
                conv = weave.converters.default

            if self.compiled_eps_loop:

                # C loop over eps, all inner loops must be compiled as well
                #
                weave.inline(self.code, self.arg_names, local_dict=arg_values,
                             extra_compile_args=compiler_args,
                             extra_link_args=linker_args,
                             type_converters=conv, compiler=self.compiler,
                             verbose=self.compiler_verbose)

            else:

                # Python loop over eps
                #
                for idx, e in enumerate(eps_arr):

                    # C loop over random dimensions
                    #
                    arg_values['e'] = e  # prepare the parameter
                    mu_q = weave.inline(self.code, self.arg_names,
                                        local_dict=arg_values,
                                        extra_compile_args=compiler_args,
                                        extra_link_args=linker_args,
                                        type_converters=conv,
                                        compiler=self.compiler,
                                        verbose=self.compiler_verbose)

                    # add the value to the return array
                    mu_q_arr[idx] = mu_q

            var_q_arr = np.zeros_like(mu_q_arr)

            return mu_q_arr, var_q_arr

        return mu_q_method

    #=========================================================================
    # Extra compiler arguments
    #=========================================================================
    use_extra = Bool(False, codegen_option=True)

    extra_args = Property(depends_on='use_extra, +codegen_option, recalc')

    @cached_property
    def _get_extra_args(self):
        if self.use_extra == True:
            # , "-fno-openmp", "-ftree-vectorizer-verbose=3"]
            compiler_args = [
                "-DNDEBUG -g -fwrapv -O3 -march=native", "-ffast-math"]
            linker_args = []  # ["-fno-openmp"]
            return compiler_args, linker_args
        elif self.use_extra == False:
            return [], []

    #=========================================================================
    # Auxiliary methods
    #=========================================================================
    def _set_compiler(self):
        '''Catch eventual mismatch between scipy.weave and compiler 
        '''
        if platform.system() == 'Linux':
            #os.environ['CC'] = 'gcc-4.1'
            #os.environ['CXX'] = 'g++-4.1'
            os.environ['OPT'] = '-DNDEBUG -g -fwrapv -O3'
        elif platform.system() == 'Windows':
            # not implemented
            pass

    def _get_code_dG_declare(self):
        '''Constant dG value - for PGrid, MCS, LHS
        '''
        return ''

    def _get_code_dG_access(self):
        '''Default access to dG array - only needed by TGrid'''
        return ''

    def _get_arg_names_dG(self):
        return []

    def _get_arg_values_dG(self):
        return {}

    def __str__(self):
        s = 'C( '
        s += 'var_eval = %s, ' % repr(self.implicit_var_eval)
        s += 'compiled_eps_loop = %s, ' % repr(self.compiled_eps_loop)
        s += 'cached_dG = %s)' % repr(self.cached_dG)
        return s

#=========================================================================
# CodeGen for regular sampling (n-embedded loops)
#=========================================================================


class CodeGenCompiledRegular(CodeGenCompiled):

    def _get_code_inner_loops(self, inner_code_str):

        code_str = ''
        # generate loops over random variables
        n_int = self.spirrid.n_int

        idx = 0
        for name, distr in zip(self.var_names, self.spirrid.tvar_lst):

            # skip the deterministic variable
            if type(distr) is float:
                continue

            # create the loop over the random variable
            #
            if self.compiled_eps_loop:
                code_str += self.ld_.LD_BEGIN_THETA_DEEP_LOOP % {
                    't': ('\t' * (idx + 1)), 's': name, 'i': n_int}
                code_str += self.LD_ACCESS_THETA % (
                    '\t' * (idx + 1), name, name, name)
            else:
                code_str += self.ld_.LD_BEGIN_THETA_DEEP_LOOP % {
                    't': ('\t' * idx), 's': name, 'i': n_int}
                code_str += self.LD_ACCESS_THETA % (
                    '\t' * idx, name, name, name)
            idx += 1

        code_str += inner_code_str

        # close the random loops
        #
        for name in self.rand_var_names:
            code_str += self.ld_.LD_END_THETA_DEEP_LOOP

        return code_str


class CodeGenCompiledTGrid(CodeGenCompiledRegular):

    def _get_arg_names_dG(self):
        arg_names = []
        if self.cached_dG:
            arg_names += ['dG_grid']
        else:
            arg_names += ['%s_dG' % name for name in self.rand_var_names]
        return arg_names

    def _get_arg_values_dG(self):
        arg_values = {}
        if self.n_rand_vars > 0:
            if self.cached_dG:
                arg_values['dG_grid'] = self.spirrid.sampling.dG
            else:
                for name, dG_arr in zip(self.rand_var_names, self.dG_arrs):
                    arg_values['%s_dG' % name] = dG_arr
        else:
            arg_values['dG_grid'] = self.spirrid.sampling.dG
        return arg_values

    def _get_code_dG_access(self):
        if self.compiled_eps_loop:
            if self.cached_dG:  # q_g - blitz matrix used to store the grid
                code_str = self.ld_.LD_DECLARE_DG_IDX % {'t': ('\t') * (self.n_rand_vars + 1)} + \
                    ','.join([self.ld_.LD_ACCESS_DG_IDX % name
                              for name in self.rand_var_names]) + \
                    self.ld_.LD_END_BRACE
            else:  # qg
                code_str = self.ld_.LD_DECLARE_DG_PTR % {'t': ('\t') * (self.n_rand_vars + 1)} + \
                    ' * '.join([self.ld_.LD_ACCESS_DG_PTR % (name, name)
                                for name in self.rand_var_names]) + \
                    ';\n'
        else:
            if self.cached_dG:  # q_g - blitz matrix used to store the grid
                code_str = self.ld_.LD_DECLARE_DG_IDX % {'t': ('\t') * self.n_rand_vars} + \
                    ','.join([self.ld_.LD_ACCESS_DG_IDX % name
                              for name in self.rand_var_names]) + \
                    self.ld_.LD_END_BRACE
            else:  # qg
                code_str = self.ld_.LD_DECLARE_DG_PTR % {'t': ('\t') * self.n_rand_vars} + \
                    ' * '.join([self.ld_.LD_ACCESS_DG_PTR % (name, name)
                                for name in self.rand_var_names]) + \
                    ';\n'
        return code_str


class CodeGenCompiledPGrid(CodeGenCompiledRegular):

    def _get_code_dG_declare(self):
        n_sim = self.spirrid.sampling.n_sim
        if self.compiled_eps_loop:
            if self.cached_dG:
                return '\t' + self.ld_.LD_ASSIGN_DG % (1.0 / n_sim)
            else:
                return '\t' + self.ld_.LD_ASSIGN_DG % (1.0 / n_sim)
        else:
            if self.cached_dG:
                return self.ld_.LD_ASSIGN_DG % (1.0 / n_sim)
            else:
                return self.ld_.LD_ASSIGN_DG % (1.0 / n_sim)

#=========================================================================
# CodeGen for irregular sampling (n-embedded loops)
#=========================================================================


class CodeGenCompiledIrregular(CodeGenCompiled):

    def _get_code_dG_declare(self):
        n_sim = self.spirrid.sampling.n_sim
        if self.compiled_eps_loop:
            if self.cached_dG:
                return '\t' + self.ld_.LD_ASSIGN_DG % (1.0 / n_sim)
            else:
                return '\t' + self.ld_.LD_ASSIGN_DG % (1.0 / n_sim)
        else:
            if self.cached_dG:
                return self.ld_.LD_ASSIGN_DG % (1.0 / n_sim)
            else:
                return self.ld_.LD_ASSIGN_DG % (1.0 / n_sim)

    def _get_code_inner_loops(self, inner_code_str):

        code_str = ''
        # generate loops over random variables
        n_int = self.spirrid.n_int
        n_sim = self.spirrid.sampling.n_sim

        if self.compiled_eps_loop:
            code_str += '\t' + self.ld_.LD_BEGIN_THETA_FLAT_LOOP % n_sim
        else:
            code_str += self.ld_.LD_BEGIN_THETA_FLAT_LOOP % n_sim

        for name, distr in zip(self.var_names, self.spirrid.tvar_lst):

            # skip the deterministic variable
            if type(distr) is float:
                continue

            # pointer access possible for single dimensional arrays
            # use the pointer arithmetics for accessing the pdfs
            if self.compiled_eps_loop:
                if self.cached_dG:
                    code_str += '\t' + \
                        self.ld_.LD_ACCESS_THETA_FLAT_IDX % (name, name)
                else:
                    code_str += '\t' + \
                        self.ld_.LD_ACCESS_THETA_FLAT_PTR % (name, name)
            else:
                if self.cached_dG:
                    code_str += self.ld_.LD_ACCESS_THETA_FLAT_IDX % (
                        name, name)
                else:
                    code_str += self.ld_.LD_ACCESS_THETA_FLAT_PTR % (
                        name, name)

        code_str += inner_code_str

        # close the random loops
        #
        code_str += self.ld_.LD_END_THETA_FLAT_LOOP

        return code_str
