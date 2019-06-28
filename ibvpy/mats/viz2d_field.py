'''
Created on Feb 11, 2018

@author: rch
'''

import copy
import os
import tempfile

from tvtk.api import tvtk
from view.plot2d import Vis2D, Viz2D

import numpy as np
import traits.api as tr


class Vis2DField(Vis2D):

    model = tr.DelegatesTo('sim')

    x_file = tr.File
    file_list = tr.List(tr.File)

    var = tr.Str('<unnamed>')

    dir = tr.Directory

    def new_dir(self):
        self.dir = tempfile.mkdtemp()

    def setup(self):
        self.new_dir()
        # make a loop over the DomainState
        fe_domain = self.sim.tstep.fe_domain
        domain = fe_domain[2]
        xdomain = domain.xdomain
        r_Eia = np.einsum(
            'Eira,Eia->Eir',
            xdomain.T_Emra[..., :xdomain.x_Eia.shape[-1]], xdomain.x_Eia
        )

        file_name = 'slice_x_%s' % (self.var,)
        target_file = os.path.join(
            self.dir, file_name.replace('.', '_') + '.npy'
        )
        #print('r', r_Eia[..., :-1])
        np.save(target_file, r_Eia[..., :-1])
        self.x_file = target_file

    def get_x_Eir(self):
        return np.load(self.x_file)

    def update(self):
        ts = self.sim.tstep
        fe_domain = self.sim.tstep.fe_domain
        domain = fe_domain[2]
        xdomain = domain.xdomain
        U = ts.U_k
        t = ts.t_n1
        s_Emr = xdomain.map_U_to_field(U)

        var_function = domain.tmodel.var_dict.get(self.var, None)
        if var_function == None:
            raise ValueError('no such variable' % self.var)

        state_k = copy.deepcopy(domain.state_n)
        var_k = var_function(s_Emr, ts.t_n1, **state_k)

        target_file = self.filename(t)

        #np.save(target_file, s_Emr)
        np.save(target_file, var_k)
        self.file_list.append(target_file)

    def filename(self, t):
        file_name = 'slice_%s_step_%008.4f' % (self.var, t)
        target_file = os.path.join(
            self.dir, file_name.replace('.', '_')
        ) + '.npy'
        return target_file


class Viz2DField(Viz2D):

    def setup(self):
        print('Viz2DField:seetup')
        pass

    adaptive_y_range = tr.Bool(True)
    initial_plot = tr.Bool(True)
    y_min = tr.Float(0)
    y_max = tr.Float(0)

    def plot(self, ax, vot, *args, **kw):
        idx = self.vis2d.sim.hist.get_time_idx(vot)
        if len(self.vis2d.file_list) <= idx:
            return
        file_name = self.vis2d.file_list[idx]
        X_Eir = self.vis2d.get_x_Eir()
        x_sorted = X_Eir[..., 0].flatten()
        var_Eir = np.load(file_name)
        slip_sorted = var_Eir.flatten()
        ax.plot(x_sorted, slip_sorted, linewidth=2, color='blue')
        ax.fill_between(x_sorted, slip_sorted,
                        facecolor='blue', alpha=0.2)
#        ax.plot([0, L], [0, 0], color='black')
        ax.set_ylabel(self.vis2d.var)
        ax.set_xlabel('bond length')
        ymin, ymax = np.min(slip_sorted), np.max(slip_sorted)

        if self.adaptive_y_range:
            if self.initial_plot:
                self.y_max = ymax
                self.y_min = ymin
                self.initial_plot = False
                return
        self.y_max = max(ymax, self.y_max)
        self.y_min = min(ymin, self.y_min)
        ax.set_ylim(ymin=self.y_min, ymax=self.y_max)
